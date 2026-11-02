import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.latent_seq import LatentSeqDataset
from models.dynamics_coarse3d import CoarseDynamics3D

MAX_HIST_NOISE, INFER_HIST_NOISE, DISCOUNT = 0.4, 0.1, 0.95


def forward(model, z_hist, act_hist, act_next, ns_hist, z_next):
    B, n = z_hist.shape[:2]
    noise = torch.cat([ns_hist, torch.ones(B, 1, device=z_hist.device)], 1)
    actions = torch.cat([act_hist, act_next], 1).permute(1, 0, 2)
    return model(torch.cat([z_hist, z_next], 1), actions, noise_std=noise)[:, -1:]


def predict_train(model, z_hist, act_hist, act_next):
    B, n, C, nz, H, W = z_hist.shape
    device = z_hist.device
    ns = torch.rand(B, n, device=device) * MAX_HIST_NOISE
    noisy = z_hist + torch.randn_like(z_hist) * ns[:, :, None, None, None, None]
    return forward(model, noisy, act_hist, act_next, ns, torch.randn(B, 1, C, nz, H, W, device=device))


@torch.no_grad()
def predict(model, z_hist, act_hist, act_next, hist_noise=INFER_HIST_NOISE):
    B, n, C, nz, H, W = z_hist.shape
    device = z_hist.device
    return forward(
        model,
        z_hist,
        act_hist,
        act_next,
        torch.full((B, n), hist_noise, device=device),
        torch.randn(B, 1, C, nz, H, W, device=device),
    )


def weighted_mse(pred, gt, w):
    sq = (pred - gt) ** 2
    return (sq * w).sum() / w.expand_as(sq).sum().clamp(min=1.0)


def validate(model, loader, n_hist, max_ar, val_iters, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= val_iters:
                break
            z, act = batch["z_seq"].to(device), batch["actions"].to(device)
            z_hist = z[:, :n_hist]
            for step in range(1 + max_ar):
                t = n_hist + step
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = predict(model, z_hist, act[:, t - n_hist : t], act[:, t : t + 1])
                pred = pred.float()
                total += F.mse_loss(pred, z[:, t : t + 1]).item()
                n += 1
                z_hist = torch.cat([z_hist, pred], 1)[:, -n_hist:]
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--val-latent-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-history", type=int, default=8)
    p.add_argument("--ar-schedule", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--ar-schedule-epochs", type=int, nargs="+", default=[50 * i for i in range(1, 11)])
    p.add_argument("--object-alpha", type=float, default=20.0)
    p.add_argument("--tactile-alpha", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--val-iters", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    device = "cuda"
    out = Path(a.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    n_hist, seq_len = a.n_history, a.n_history + 1 + max(a.ar_schedule)
    norm = out / "action_norm_params.npz"
    if not norm.exists():
        mean, std = LatentSeqDataset(a.latent_dir, seq_len).compute_action_norm_params()
        np.savez(norm, action_mean=mean, action_std=std)
    d = np.load(norm)
    amean, astd = d["action_mean"], d["action_std"]
    train_ds = LatentSeqDataset(a.latent_dir, seq_len, action_mean=amean, action_std=astd)
    val_ds = LatentSeqDataset(a.val_latent_dir, seq_len, action_mean=amean, action_std=astd)
    train_loader = DataLoader(
        train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, drop_last=True)
    C = train_ds[0]["z_seq"].shape[1]
    raw = CoarseDynamics3D(
        latent_channels=C,
        action_dim=train_ds.action_dim,
        n_history=n_hist,
        n_future=1,
        width=56,
        action_emb_dim=128,
        n_levels=3,
        film=True,
    ).to(device)
    model = torch.compile(raw)
    opt = torch.optim.AdamW(raw.parameters(), lr=a.lr, weight_decay=0.0001)
    start, best = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        raw.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start, best = ck["epoch"] + 1, ck["best_val"]
    elif a.init:
        raw.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model_state_dict"])
    for epoch in range(start, a.ar_schedule_epochs[-1]):
        model.train()
        max_ar = next((ar for end, ar in zip(a.ar_schedule_epochs, a.ar_schedule) if epoch < end), a.ar_schedule[-1])
        total, it = 0.0, iter(train_loader)
        for _ in range(a.iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            z, act = batch["z_seq"].to(device, non_blocking=True), batch["actions"].to(device, non_blocking=True)
            w = 1.0 + (a.object_alpha - 1.0) * batch["obj_occ"].to(device, non_blocking=True)
            if "tac_occ" in batch and a.tactile_alpha != 1.0:
                w = w * (1.0 + (a.tactile_alpha - 1.0) * batch["tac_occ"].to(device, non_blocking=True))
            opt.zero_grad()
            z_hist = z[:, :n_hist].clone()
            for ar in range(1 + max_ar):
                t = n_hist + ar
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = predict_train(model, z_hist, act[:, t - n_hist : t], act[:, t : t + 1])
                pred = pred.float()
                loss = DISCOUNT**ar * weighted_mse(pred[:, 0], z[:, t], w[:, t].unsqueeze(1))
                loss.backward()
                total += loss.item()
                z_hist = torch.cat([z_hist, pred.detach()], 1)[:, -n_hist:]
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 10.0)
            opt.step()
        val = validate(raw, val_loader, n_hist, max_ar, a.val_iters, device)
        print(f"[ep {epoch}] ar {max_ar} train {total / a.iters:.5f} val {val:.5f}", flush=True)
        ck = dict(
            model_state_dict=raw.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            epoch=epoch,
            best_val=best,
            action_mean=amean,
            action_std=astd,
            args=vars(a),
        )
        torch.save(ck, out / "checkpoints" / "latest.pth")
        if val < best:
            best = ck["best_val"] = val
            torch.save(ck, out / "checkpoints" / "best.pth")
        if (epoch + 1) % a.save_interval == 0:
            torch.save(ck, out / "checkpoints" / f"model_{epoch}.pth")


if __name__ == "__main__":
    main()
