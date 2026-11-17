"""Synthetic X-ray data: STL part + random pores -> 6 freehand views rendered with gVXR.

Usage: python synth.py [part.stl] [n_samples]
Writes data/sample_XXXX.npz and self-checks every sample.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation

# Fixed bench geometry (mm). Source on -y, detector on +y; the part pose moves, not the bench.
GEOM = dict(sod=400.0, sdd=600.0, w=384, h=384, px=0.4, kev=80.0, element="Al")
N_VIEWS = 6
MASK_THR = 5e-3  # min ln(I_defect/I_nominal); ~0.1 mm of Al at 80 keV


def make_part():
    """Default test part: 60x40x30 mm block, two unequal through-holes and a corner notch.
    Deliberately asymmetric: a symmetric part has several poses giving the same X-ray."""
    part = trimesh.creation.box([60, 40, 30])
    cuts = [trimesh.creation.cylinder(4, 50, sections=64), trimesh.creation.cylinder(2.5, 50, sections=64),
            trimesh.creation.box([14, 50, 10])]
    for c, at in zip(cuts, ([-15, 0, 0], [16, 6, 0], [27, 0, 13])):
        c.apply_translation(at)
        part = part.difference(c, engine="manifold")
    return part


def random_pore(rng, center):
    """Perturbed ellipsoid: icosphere, random axes/orientation, low-frequency radial bumps."""
    s = trimesh.creation.icosphere(3)
    v = s.vertices
    bumps = sum(rng.normal(0, 0.08) * np.sin(rng.uniform(1, 3) * v @ rng.normal(size=3) + rng.uniform(0, 6))
                for _ in range(4))
    v = v * (1 + bumps)[:, None] * rng.uniform(1.5, 4.0, 3)
    v = Rotation.random(random_state=rng).apply(v) + center
    return trimesh.Trimesh(v, s.faces)


def place_pores(rng, part, n, margin=1.5, tries=200):
    pores = []
    lo, hi = part.bounds
    for _ in range(tries):
        if len(pores) == n:
            break
        p = random_pore(rng, rng.uniform(lo, hi))
        if trimesh.proximity.signed_distance(part, p.vertices).min() < margin:
            continue  # trimesh: positive = inside
        if any(np.linalg.norm(p.centroid - q.centroid) < p.bounding_sphere.primitive.radius
               + q.bounding_sphere.primitive.radius + margin for q in pores):
            continue
        pores.append(p)
    return pores


def random_pose(rng):
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-5, 5, 3)
    return T


def project(pts, pose, g=GEOM):
    """Part-frame points (N,3) -> pixel (row, col), matching gVXR's detector layout."""
    p = pts @ pose[:3, :3].T + pose[:3, 3]
    m = g["sdd"] / (p[:, 1] + g["sod"])
    col = (g["w"] - 1) / 2 - p[:, 0] * m / g["px"]
    row = (g["h"] - 1) / 2 - p[:, 2] * m / g["px"]
    return np.stack([row, col], 1)


class Renderer:
    def __init__(self, g=GEOM):
        from gvxrPython3 import gvxr
        self.gvxr, self.g = gvxr, g
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(0, -g["sod"], 0, "mm")
        gvxr.usePointSource()
        gvxr.setMonoChromaticPerPixelAtSDD(g["kev"], "keV", 1000)
        gvxr.setDetectorPosition(0, g["sdd"] - g["sod"], 0, "mm")
        gvxr.setDetectorUpVector(0, 0, -1)
        gvxr.setDetectorNumberOfPixels(g["w"], g["h"])
        gvxr.setDetectorPixelSize(g["px"], g["px"], "mm")
        self.tmp = Path(tempfile.mkdtemp())

    def __call__(self, mesh, pose):
        """Flat-field-normalised transmission image I/I0 of mesh at pose."""
        f = self.tmp / "m.stl"
        mesh.copy().apply_transform(pose).export(f)
        gv = self.gvxr
        gv.removePolygonMeshesFromSceneGraph()
        gv.loadMeshFile("part", str(f), "mm")
        gv.setElement("part", self.g["element"])
        return (np.array(gv.computeXRayImage()) / gv.getTotalEnergyWithDetectorResponse()).astype(np.float32)


def make_sample(rng, part, render):
    pores = place_pores(rng, part, rng.integers(1, 4))
    defect = part.difference(trimesh.util.concatenate(pores), engine="manifold")
    poses = np.stack([random_pose(rng) for _ in range(N_VIEWS)])
    img = np.stack([render(defect, T) for T in poses])
    nom = np.stack([render(part, T) for T in poses])
    out = dict(images=img, nominal=nom, masks=np.log(img / nom) > MASK_THR, poses=poses,
               part_v=part.vertices, part_f=part.faces, defect_volume=defect.volume,
               n_pores=len(pores), **{k: np.array(v) for k, v in GEOM.items()})
    for i, p in enumerate(pores):
        out[f"pore{i}_v"], out[f"pore{i}_f"] = p.vertices, p.faces
    return out


def load_pores(d):
    return [trimesh.Trimesh(d[f"pore{i}_v"], d[f"pore{i}_f"]) for i in range(int(d["n_pores"]))]


def check(d):
    pores = load_pores(d)
    part = trimesh.Trimesh(d["part_v"], d["part_f"])
    lost = part.volume - float(d["defect_volume"])
    want = sum(p.volume for p in pores)
    assert abs(lost - want) < 0.01 * want, (lost, want)
    for T, m in zip(d["poses"], d["masks"]):
        for p in pores:
            r, c = np.round(project(p.centroid[None], T)[0]).astype(int)
            assert 0 <= r < m.shape[0] and 0 <= c < m.shape[1], "pore projects off detector"
            assert m[r, c], "pore centroid not inside its mask"


if __name__ == "__main__":
    part = trimesh.load(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else make_part()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    assert part.is_watertight, "part STL must be watertight"
    out = Path("data"); out.mkdir(exist_ok=True)
    rng, render = np.random.default_rng(0), Renderer()
    for i in range(n):
        d = make_sample(rng, part, render)
        check(d)
        np.savez_compressed(out / f"sample_{i:04d}.npz", **d)
    print(f"ok: {n} samples in {out}/")
