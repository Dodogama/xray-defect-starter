"""2D pore segmentation on raw X-rays: U-Net trained on synth.py data.

Usage: python synth.py "" 300 && python segment.py
Saves seg.pt; self-check: held-out Dice > 0.8 on noisy images.
"""
from pathlib import Path
import numpy as np, torch, albumentations as A, segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

DEV = "cuda"


def degrade(img, rng, n0=None, haze=None):
    """Physics-ish corruption of transmission I/I0: scatter haze + photon (Poisson) noise."""
    n0 = rng.uniform(2e3, 5e4) if n0 is None else n0
    haze = rng.uniform(0, 0.15) if haze is None else haze
    img = img + haze * gaussian_filter(img, 30)
    return rng.poisson(img * n0) / n0


def to_input(img):
    """Transmission -> normalised line-integral image, the network's only input."""
    x = -np.log(np.clip(img, 1e-4, 1))  # noise/haze push air above 1; negatives NaN under RandomGamma
    return (x / (x.max() + 1e-6)).astype(np.float32)


AUG = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
                 A.RandomGamma((70, 150)), A.GaussianBlur((3, 5), p=0.3)])


class Views(torch.utils.data.Dataset):
    def __init__(self, files, train):
        self.img, self.mask, self.train = [], [], train
        for f in files:
            d = np.load(f)
            self.img += list(d["images"]); self.mask += list(d["masks"])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        x = to_input(degrade(self.img[i], rng) if self.train else degrade(self.img[i], rng, 1e4, 0.05))
        m = self.mask[i].astype(np.float32)
        if self.train:
            a = AUG(image=x, mask=m); x, m = a["image"], a["mask"]
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(m))[None]


def model():
    return smp.Unet("resnet18", encoder_weights="imagenet", in_channels=1, classes=1)


@torch.no_grad()
def predict(net, images):
    """(V,H,W) transmission images -> (V,H,W) bool pore masks."""
    x = torch.from_numpy(np.stack([to_input(i) for i in images]))[:, None].to(DEV)
    return (net(x).sigmoid() > 0.5).squeeze(1).cpu().numpy()


def dice(net, loader):
    inter = total = 0
    with torch.no_grad():
        for x, m in loader:
            p = (net(x.to(DEV)).sigmoid() > 0.5).float()
            inter += (p * m.to(DEV)).sum().item(); total += (p.sum() + m.sum()).item()
    return 2 * inter / total


if __name__ == "__main__":
    files = sorted(Path("data").glob("sample_*.npz"))
    split = int(0.8 * len(files))  # split by sample so no view of a val pore is seen in training
    dl = lambda f, train: torch.utils.data.DataLoader(Views(f, train), batch_size=8, shuffle=train, num_workers=4)
    train, val = dl(files[:split], True), dl(files[split:], False)
    net = model().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    epochs = 15
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * len(train))
    loss_fn = lambda p, m: smp.losses.DiceLoss("binary")(p, m) + smp.losses.SoftBCEWithLogitsLoss()(p, m)
    for ep in range(epochs):
        net.train()
        for x, m in train:
            with torch.autocast(DEV, torch.bfloat16):
                loss = loss_fn(net(x.to(DEV)), m.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        print(f"epoch {ep} loss {loss.item():.3f} val dice {dice(net, val):.3f}", flush=True)
    torch.save(net.state_dict(), "seg.pt")
    d = dice(net, val)
    assert d > 0.8, f"val dice {d:.3f} <= 0.8"
    print(f"ok: val dice {d:.3f}, saved seg.pt")
