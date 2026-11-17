"""Reconstruct each pore as a smooth 3D surface from ~6 posed X-rays, refining each view's pose jointly.

Pore = star-shaped surface: centre c, radius r(u) = exp(poly_deg4(u)) over unit directions u (same span
as spherical harmonics to degree 4). Watertight and smooth by construction -> no jagged geometry.
Data term: residual r = ln(I / I_nominal) / mu = mm of missing material along each ray.
Usage: python recon.py   (self-checks on data/sample_0000.npz)
"""
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np, torch, trimesh
from scipy import ndimage
from synth import GEOM, project, load_pores
from register import DEV, skew_exp

DEG = 4
POWERS = torch.tensor([[e.count(i) for i in range(3)] for d in range(DEG + 1)
                       for e in combinations_with_replacement(range(3), d)], device=DEV).float()  # (35,3)
ORDER = POWERS.sum(1)


def mu_per_mm(g=GEOM):
    from gvxrPython3 import gvxr
    el = str(g["element"])
    return gvxr.getMassAttenuationFromElement(el, float(g["kev"]), "keV") * gvxr.getDensityFromElement(el) / 10


def pixel_rays(rows, cols, g=GEOM):
    """Source point and unit ray directions (world frame) through pixel centres."""
    det = np.stack([((g["w"] - 1) / 2 - cols) * g["px"], np.full(len(rows), g["sdd"] - g["sod"]),
                    ((g["h"] - 1) / 2 - rows) * g["px"]], 1)
    o = np.array([0.0, -g["sod"], 0.0])
    d = det - o
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


def to_part(o, d, T):
    R, t = T[:3, :3], T[:3, 3]
    return (o - t) @ R, d @ R  # R^T applied to row vectors


def triangulate(origins, dirs):
    """Least-squares closest point to a set of rays."""
    A = sum(np.eye(3) - np.outer(d, d) for d in dirs)
    b = sum((np.eye(3) - np.outer(d, d)) @ o for o, d in zip(origins, dirs))
    return np.linalg.solve(A, b)


def init_pores(masks, poses, g=GEOM, min_views=None):
    """Match mask blobs across views by triangulating pairs and checking every view; return (centre, radius)."""
    min_views = min_views or len(masks) - 1
    blobs = []
    for m, T in zip(masks, poses):
        lab, n = ndimage.label(m)
        cen = ndimage.center_of_mass(m, lab, range(1, n + 1))
        area = ndimage.sum(m, lab, range(1, n + 1))
        blobs.append((lab, np.array(cen).reshape(-1, 2), np.array(area)))
    a, b = np.argsort([-len(x[1]) for x in blobs])[:2]  # two views with the most blobs
    found = []
    for ca in blobs[a][1]:
        for cb in blobs[b][1]:
            rays = [to_part(*pixel_rays(*c[:, None], g=g), poses[v]) for v, c in ((a, ca), (b, cb))]
            p = triangulate([r[0] for r in rays], [r[1][0] for r in rays])
            hits = []
            for v, (lab, cen, area) in enumerate(blobs):
                r, c = np.round(project(p[None], poses[v], g)[0]).astype(int)
                if 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c]:
                    hits.append((v, lab[r, c] - 1))
            if len(hits) >= min_views and all(np.linalg.norm(p - q) > 1.0 for q, _ in found):
                found.append((p, hits))
    pores = []
    for p, hits in found:
        # refine with blobs that only this pore claims (merged blobs have biased centroids)
        own = [(v, k) for v, k in hits if sum((v, k) in h for _, h in found) == 1]
        if len(own) >= 2:
            rays = [to_part(*pixel_rays(*blobs[v][1][k][:, None], g=g), poses[v]) for v, k in own]
            p = triangulate([r[0] for r in rays], [r[1][0] for r in rays])
        m = g["sdd"] / (np.array([(poses[v][:3, :3] @ p + poses[v][:3, 3])[1] for v, _ in own or hits]) + g["sod"])
        radius = np.mean([np.sqrt(blobs[v][2][k] / np.pi) * g["px"] / mv for (v, k), mv in zip(own or hits, m)])
        pores.append((p, radius))
    return pores


def roi_pixels(pores, poses, shape, g=GEOM, pad=2.0):
    """Per view, all pixels within the projected (padded) bounding sphere of any pore."""
    out = []
    rr, cc = np.indices(shape)
    for T in poses:
        keep = np.zeros(shape, bool)
        for c, r in pores:
            pr, pc = project(c[None], T, g)[0]
            m = g["sdd"] / ((T[:3, :3] @ c + T[:3, 3])[1] + g["sod"])
            keep |= np.hypot(rr - pr, cc - pc) < (pad * r + 1.0) * m / g["px"]
        out.append(np.nonzero(keep))
    return out


def radius(coef, u):
    """Star-shaped radius r(u) = exp(sum coef * monomials(u)); u (...,3), coef broadcastable to (...,35)."""
    pw = u[..., None] ** torch.arange(DEG + 1, device=DEV)  # (...,P,3,DEG+1) per-axis power tables
    p = POWERS.long()
    feats = pw[..., 0, p[:, 0]] * pw[..., 1, p[:, 1]] * pw[..., 2, p[:, 2]]  # (...,P,35)
    return torch.exp((feats * coef).sum(-1))


class PoreModel(torch.nn.Module):
    def __init__(self, init, n_views):
        super().__init__()
        c = torch.tensor(np.array([p for p, _ in init]), device=DEV).float()
        r = torch.tensor([r for _, r in init], device=DEV).float()
        self.c = torch.nn.Parameter(c.clone())
        coef = torch.zeros(len(init), len(POWERS), device=DEV)
        coef[:, 0] = r.log()
        self.coef = torch.nn.Parameter(coef)
        self.w = torch.nn.Parameter(torch.zeros(n_views, 3, device=DEV))   # per-view pose correction
        self.dt = torch.nn.Parameter(torch.zeros(n_views, 3, device=DEV))
        self.c0, self.rho = c, 2.0 * r + 1.0  # fixed sampling spheres

    def forward(self, o, d, view, R, t, n=128, tau=0.1):
        """Missing material (mm) along world rays o + s d of view index `view`, summed over pores."""
        Rv = R[view] @ skew_exp(self.w[view])
        op = torch.einsum("nji,nj->ni", Rv, o - t[view] - self.dt[view])
        dp = torch.einsum("nji,nj->ni", Rv, d)
        oc = op[:, None] - self.c0  # (N,P,3)
        s_mid = -(oc * dp[:, None]).sum(-1)  # closest approach to each sampling sphere
        s = s_mid[..., None] + self.rho[:, None] * torch.linspace(-1, 1, n, device=DEV)  # (N,P,n)
        x = op[:, None, None] + s[..., None] * dp[:, None, None]  # (N,P,n,3)
        v = x - self.c[:, None]
        dist = v.norm(dim=-1).clamp_min(1e-6)
        occ = torch.sigmoid((radius(self.coef[:, None], v / dist[..., None]) - dist) / tau)
        return (occ.sum(-1) * (2 * self.rho / (n - 1))).sum(-1)

    def meshes(self, sub=4):
        s = trimesh.creation.icosphere(sub)
        u = torch.tensor(s.vertices, device=DEV).float()[:, None].expand(-1, len(self.c), -1)
        with torch.no_grad():
            rad = radius(self.coef, u).cpu().numpy()  # (V,P)
        c = self.c.detach().cpu().numpy()
        return [trimesh.Trimesh(s.vertices * rad[:, i:i + 1] + c[i], s.faces) for i in range(len(c))]


def reconstruct(images, nominal, masks, poses, refine_pose=True, iters=(400, 400), g=GEOM, log=False):
    mu = mu_per_mm(g)
    init = init_pores(masks, poses, g)
    assert init, "no pore found in enough views"
    rois = roi_pixels(init, poses, masks.shape[1:], g)
    o_l, d_l, v_l, y_l = [], [], [], []
    for v, ((rows, cols), T) in enumerate(zip(rois, poses)):
        o, d = pixel_rays(rows, cols, g=g)
        o_l.append(np.broadcast_to(o, d.shape)); d_l.append(d); v_l.append(np.full(len(d), v))
        y_l.append(np.log(images[v][rows, cols] / nominal[v][rows, cols]) / mu)
    f = lambda a: torch.tensor(np.concatenate(a), device=DEV).float()
    o, d, y = f(o_l), f(d_l), f(y_l)
    view = torch.tensor(np.concatenate(v_l), device=DEV)
    R = torch.tensor(poses[:, :3, :3], device=DEV).float()
    t = torch.tensor(poses[:, :3, 3], device=DEV).float()
    model = PoreModel(init, len(poses))
    smooth = 1e-3 * ORDER ** 2  # penalise high-order wiggles
    for phase, n_it in enumerate(iters):
        joint = phase == 1 and refine_pose
        params = [{"params": [model.c], "lr": 0.02}, {"params": [model.coef], "lr": 0.02}]
        if joint:
            params += [{"params": [model.w], "lr": 1e-3}, {"params": [model.dt], "lr": 0.01}]
        opt = torch.optim.Adam(params)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_it)
        for it in range(n_it):
            b = torch.randint(len(y), (min(4096, len(y)),), device=DEV)  # stochastic ray batch
            loss = torch.nn.functional.huber_loss(model(o[b], d[b], view[b], R, t), y[b], delta=0.5)
            loss = loss + (smooth[1:] * model.coef[:, 1:] ** 2).sum()
            if joint:  # registration is good to ~1 px (~0.3 mm, ~0.5 deg): keep corrections near that
                loss = loss + 1e-3 * ((model.dt / 0.3) ** 2).sum() + 1e-3 * ((model.w / np.radians(0.5)) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if log and it % 100 == 0:
                print(f"phase {phase} it {it} loss {loss.item():.4f}", flush=True)
    return model.meshes(), model


def chamfer(a, b, n=5000):
    pa, pb = a.sample(n), b.sample(n)
    return (trimesh.proximity.closest_point(b, pa)[1].mean() + trimesh.proximity.closest_point(a, pb)[1].mean()) / 2


def inspect(pores, part):
    """Inspection report per pore, in the part frame."""
    rows = []
    for p in pores:
        rows.append(dict(centroid=np.round(p.centroid, 2), volume_mm3=round(p.volume, 2),
                         eq_diameter_mm=round((6 * p.volume / np.pi) ** (1 / 3), 2),
                         min_wall_mm=round(trimesh.proximity.signed_distance(part, p.vertices).min(), 2)))
    return rows


def score(rec, gt):
    """Match reconstructed pores to ground truth by centroid; return per-pore (chamfer/diam, vol err)."""
    out = []
    for p in gt:
        q = min(rec, key=lambda q: np.linalg.norm(q.centroid - p.centroid))
        diam = (6 * p.volume / np.pi) ** (1 / 3)
        out.append((chamfer(q, p) / diam, abs(q.volume - p.volume) / p.volume))
    return out


if __name__ == "__main__":
    d = np.load("data/sample_0000.npz")
    gt, part = load_pores(d), trimesh.Trimesh(d["part_v"], d["part_f"])
    assert d["images"].shape[0] == 6

    # Check 1: true poses, true masks, true nominal renders
    rec, _ = reconstruct(d["images"], d["nominal"], d["masks"], d["poses"], refine_pose=False)
    s = score(rec, gt)
    print("check 1 (true poses): chamfer/diam, vol err =", [(round(a, 3), round(b, 3)) for a, b in s])
    assert all(a < 0.1 and b < 0.10 for a, b in s), "reconstruction with true poses not accurate"
    for r in inspect(rec, part):
        print("  ", r)
    print("ok: check 1")

    # Check 2: the realistic chain - noisy images, U-Net masks, registered poses, nominal rendered at them
    from register import register, voxelize, pose_error
    from segment import degrade, model as seg_model, predict
    from synth import Renderer
    cache = Path("data/sample_0000_reg.npy")
    if not cache.exists():
        vol, lo, hi = voxelize(part)
        np.save(cache, np.stack([register(vol, lo, hi, img)[0] for img in d["images"]]))
    poses = np.load(cache)
    for T, Tt in zip(poses, d["poses"]):
        print("  registered: rot %.3f deg, reproj %.3f px, depth %+.2f mm" % pose_error(T, Tt, part.vertices))
    rng = np.random.default_rng(0)
    noisy = np.stack([degrade(i, rng, 5e4, 0.0) for i in d["images"]]).astype(np.float32)
    net = seg_model().to(DEV); net.load_state_dict(torch.load("seg.pt")); net.eval()
    masks = predict(net, noisy)
    render = Renderer()
    nominal = np.stack([render(part, T) for T in poses])
    res = {}
    for refine in (False, True):
        torch.manual_seed(0)
        rec, _ = reconstruct(noisy, nominal, masks, poses, refine_pose=refine)
        res[refine] = score(rec, gt)
        print(f"check 2 (registered, refine_pose={refine}): chamfer/diam, vol err =",
              [(round(a, 3), round(b, 3)) for a, b in res[refine]])
    frozen, joint = (np.mean([a for a, _ in res[k]]) for k in (False, True))
    assert joint < frozen, f"pose refinement did not help: {joint:.4f} vs {frozen:.4f}"
    print(f"ok: check 2, mean chamfer/diam {frozen:.4f} frozen -> {joint:.4f} with pose refinement")
