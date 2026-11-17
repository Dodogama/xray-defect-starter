"""6-DoF 2D/3D registration of each X-ray to the nominal STL.

Differentiable DRR = trilinear ray-marching through the voxelised nominal part (torch grid_sample),
same bench geometry as synth.GEOM. Freehand poses -> multi-start over SO(3), then coarse-to-fine refine.
Usage: python register.py   (self-check on data/sample_0000.npz)
"""
import numpy as np, torch, trimesh
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from synth import GEOM, project

DEV = "cuda"


def voxelize(mesh, pitch=0.15):
    """Occupancy of voxel centres (D,H,W indexed z,y,x) + bounds. Exact inside test, so the trilinear
    0.5 level sits on the true surface (trimesh.voxelized keeps every surface voxel: ~1% too big, which
    biases registration depth by ~4 mm). One +y ray per (x,z) column; inside = odd hits below."""
    lo = mesh.bounds[0] - 2 * pitch + 1e-4 * np.pi  # jitter off axis-aligned faces/edges
    n = np.ceil((mesh.bounds[1] - mesh.bounds[0]) / pitch).astype(int) + 5
    x, y, z = (lo[i] + pitch * np.arange(n[i]) for i in range(3))
    X, Z = np.meshgrid(x, z, indexing="ij")
    org = np.stack([X.ravel(), np.full(X.size, y[0] - 1), Z.ravel()], 1)
    hit, ray, _ = mesh.ray.intersects_location(org, np.tile([0.0, 1.0, 0.0], (len(org), 1)), multiple_hits=True)
    occ = np.zeros((X.size, n[1]), np.float32)
    for r in np.unique(ray):
        ys = np.unique(np.round(hit[ray == r, 1], 6))  # dedupe hits on shared triangle edges
        occ[r] = np.searchsorted(ys, y) % 2
    occ = occ.reshape(n[0], n[2], n[1])  # (x,z,y)
    hi = lo + (n - 1) * pitch
    vol = torch.from_numpy(occ.transpose(1, 2, 0).copy())[None, None].to(DEV)
    return vol, torch.tensor(lo, device=DEV).float(), torch.tensor(hi, device=DEV).float()


def rays(k, g=GEOM, half_depth=60.0, n=256):
    """Ray sample points (h/k, w/k, n, 3) in the world frame, one ray per k*k detector block,
    restricted to |y| < half_depth around the rotation centre, plus the per-sample step length."""
    h, w = g["h"] // k, g["w"] // k
    row = torch.arange(h, device=DEV) * k + (k - 1) / 2
    col = torch.arange(w, device=DEV) * k + (k - 1) / 2
    r, c = torch.meshgrid(row, col, indexing="ij")
    det = torch.stack([((g["w"] - 1) / 2 - c) * g["px"], torch.full_like(r, g["sdd"] - g["sod"]),
                       ((g["h"] - 1) / 2 - r) * g["px"]], -1)
    src = torch.tensor([0.0, -g["sod"], 0.0], device=DEV)
    d = det - src
    ys = torch.linspace(-half_depth, half_depth, n, device=DEV)
    s = (ys[:, None] + g["sod"]) / d[..., None, 1:2]  # (h,w,n,1) ray parameter where y = ys
    pts = src + s * d[..., None, :]  # (h,w,n,3)
    step = d.norm(dim=-1) * (s[..., 1, 0] - s[..., 0, 0])  # (h,w)
    return pts, step


def drr(vol, lo, hi, R, t, pts, step):
    """Line-integral images (B,h,w) of the volume posed by (R (B,3,3), t (B,3)): world = R @ part + t."""
    p = torch.einsum("bij,bhwnj->bhwni", R.transpose(1, 2), pts[None] - t[:, None, None, None])
    grid = 2 * (p - lo) / (hi - lo) - 1  # x,y,z -> grid_sample's (W,H,D) order already
    v = F.grid_sample(vol.expand(len(R), -1, -1, -1, -1), grid, align_corners=True).squeeze(1)
    return v.sum(-1) * step


def target(img, k):
    """Measured transmission (H,W) -> line-integral image pooled to the k*k ray grid."""
    x = -torch.log(torch.as_tensor(img, device=DEV).clamp_min(1e-4))
    return F.avg_pool2d(x[None, None], k)[0, 0] if k > 1 else x


def ncc(a, b):
    a = a - a.mean((-2, -1), keepdim=True); b = b - b.mean((-2, -1), keepdim=True)
    return (a * b).sum((-2, -1)) / (a.norm(dim=(-2, -1)) * b.norm(dim=(-2, -1)) + 1e-8)


def skew_exp(w):
    K = torch.zeros(len(w), 3, 3, device=DEV)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    return torch.linalg.matrix_exp(K - K.transpose(1, 2))


def refine(vol, lo, hi, R0, t0, img, k, iters, lr=0.02, n=256):
    """Gradient refinement of a batch of poses: R = R0 exp([w]x), t = t0 + dt."""
    pts, step = rays(k, n=n)
    y = target(img, k)
    w = torch.zeros(len(R0), 3, device=DEV, requires_grad=True)
    dt = torch.zeros(len(R0), 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr}, {"params": [dt], "lr": lr * 20}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(iters, 1))
    for _ in range(iters):
        loss = (1 - ncc(drr(vol, lo, hi, R0 @ skew_exp(w), t0 + dt, pts, step), y)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        R, t = R0 @ skew_exp(w), t0 + dt
        return R, t, ncc(drr(vol, lo, hi, R, t, pts, step), y)


def register(vol, lo, hi, img, n_starts=2048, seed=0):
    """Freehand pose of one view. Raw NCC at an unrefined start ranks poorly (a start 10 deg from the
    truth ranked ~100th), so: score all starts, briefly refine the top 256 coarsely, then narrow."""
    R0 = torch.tensor(Rotation.random(n_starts, random_state=seed).as_matrix(), device=DEV).float()
    t0 = torch.zeros(n_starts, 3, device=DEV)
    pts, step = rays(16, n=96)
    y = target(img, 16)
    with torch.no_grad():
        score = torch.cat([ncc(drr(vol, lo, hi, R0[i:i + 64], t0[i:i + 64], pts, step), y)
                           for i in range(0, n_starts, 64)])
    best = score.topk(256).indices
    R, t, s = refine(vol, lo, hi, R0[best], t0[best], img, 16, 40, lr=0.05, n=96)
    best = s.topk(16).indices
    R, t, s = refine(vol, lo, hi, R[best], t[best], img, 8, 150, lr=0.03)
    # Near-symmetric parts have flipped poses that look identical when coarse: carry the top few plus
    # the best one's 180-degree flips about each part axis (about the volume centre) to finer res.
    top = s.topk(4).indices
    flips = torch.tensor(np.stack([np.diag(f) for f in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])]),
                         device=DEV).float()
    c = (lo + hi) / 2
    Rb, tb = R[top[0]], t[top[0]]
    Rf = Rb @ flips
    tf = tb + Rb @ c - Rf @ c
    R, t, s = refine(vol, lo, hi, torch.cat([R[top], Rf]), torch.cat([t[top], tf]), img, 4, 150, lr=0.01)
    # Final polish: NCC has ~1 deg local optima at full res, so restart from 7 jittered copies too.
    i = s.argmax()
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(8, 3, device=DEV, generator=g) * np.radians(1.0)
    w[0] = 0
    R = R[i] @ skew_exp(w)
    t = t[i] + torch.randn(8, 3, device=DEV, generator=g) * 0.5 * (w.norm(dim=1, keepdim=True) > 0)
    out = [refine(vol, lo, hi, R[j:j + 4], t[j:j + 4], img, 2, 200, lr=0.005) for j in (0, 4)]  # 2 GB peak
    R, t, s = (torch.cat(x) for x in zip(*out))
    i = s.argmax()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R[i].cpu().numpy(), t[i].cpu().numpy()
    return T, s[i].item()


def pose_error(T, T_true, pts):
    """(rotation err deg, mean reprojection err px of pts, depth err mm). Depth along the beam is
    nearly unobservable from one view (4 mm ~ 1% magnification), so reprojection is what matters."""
    dR = Rotation.from_matrix(T[:3, :3] @ T_true[:3, :3].T).magnitude()
    reproj = np.linalg.norm(project(pts, T) - project(pts, T_true), axis=1).mean()
    return np.degrees(dR), reproj, T[1, 3] - T_true[1, 3]


if __name__ == "__main__":
    d = np.load("data/sample_0000.npz")
    part = trimesh.Trimesh(d["part_v"], d["part_f"])
    vol, lo, hi = voxelize(part)
    # registration uses the nominal-part images; pores are small enough not to matter, but test on real ones
    for v, (img, T_true) in enumerate(zip(d["images"], d["poses"])):
        T, s = register(vol, lo, hi, img)
        er, ep, ed = pose_error(T, T_true, part.vertices)
        print(f"view {v}: ncc {s:.4f}  rot {er:.3f} deg  reproj {ep:.3f} px  depth {ed:+.2f} mm", flush=True)
        assert er < 0.5 and ep < 1.5, "registration did not converge"  # ~1 px floor = voxel-vs-mesh projector mismatch
    print("ok: all views registered")
