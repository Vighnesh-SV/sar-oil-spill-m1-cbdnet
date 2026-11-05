"""M1 CBD-Net: training on the official SOS split (reproduction) or on a group-disjoint split (C1).

Faithful to the official train.py: CBDNet network and losses imported unchanged from
third_party/CBDNet-main; Adam lr 2e-4; batch 4; up to 80 epochs; best checkpoint chosen on
training loss; lr /5 after >3 non-improving epochs (reloading best weights); stop after >6.
Documented deviations: shuffle=True (official False); seeds fixed; per-epoch CSV log.

usage (from experiments/):  python M1_CBDNet/train_m1.py --sensor sentinel --loss dicebce --seed 0
       group split (C1):    ... --split leakage_check/splits/sos_sentinel_group_s0.csv
                            (trains on partitions train+val+calib = 3,358 tiles, about the official 3,354)
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OFFICIAL = os.path.join(EXP, "third_party", "CBDNet-main")
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.join(EXP, "common"))
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "torch"))

from networks.CBDNet import CBDNet          # noqa: E402  (official)
from dice_bce_loss import dice_bce_loss      # noqa: E402  (official)
from boundary_bce_loss2 import jointloss     # noqa: E402  (official, paper's boundary loss)
from sos_data import SOSTrain, list_from_split_csv, list_split  # noqa: E402


def seed_worker(_):
    # each worker/epoch gets a distinct numpy seed derived from torch's per-worker seed
    np.random.seed(torch.initial_seed() % 2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="sentinel", choices=["sentinel", "palsar"])
    ap.add_argument("--loss", default="dicebce", choices=["dicebce", "joint"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N tiles")
    ap.add_argument("--split", default="", help="group-split CSV; empty = official split")
    ap.add_argument("--partitions", default="train,val,calib", help="partitions to train on (with --split)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    name = (f"M1_cbdnet_{a.sensor}_{a.loss}_s{a.seed}" + ("_groupsplit" if a.split else "")
            + (f"_debug{a.limit}" if a.limit else ""))
    out = os.path.join(EXP, "runs", name)
    os.makedirs(out, exist_ok=True)

    items = (list_from_split_csv(os.path.join(EXP, a.split), a.partitions.split(",")) if a.split
             else list_split("train", a.sensor))
    if a.limit:
        items = items[: a.limit]
    dl = torch.utils.data.DataLoader(
        SOSTrain(items), batch_size=a.bs, shuffle=True, num_workers=a.workers,
        pin_memory=True, drop_last=False, persistent_workers=a.workers > 0,
        worker_init_fn=seed_worker)

    dev = torch.device("cuda")
    net = CBDNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    crit = dice_bce_loss() if a.loss == "dicebce" else jointloss()
    ckpt = os.path.join(out, "best.pth")

    print(f"{name}: {len(items)} train tiles, {len(dl)} iters/epoch", flush=True)
    log = open(os.path.join(out, "train_log.csv"), "w", newline="")
    w = csv.writer(log); w.writerow(["epoch", "train_loss", "lr", "seconds", "saved"])
    best, no_optim, lr, tic = 100.0, 0, a.lr, time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); tot = 0.0
        for img, mask in dl:
            img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            opt.zero_grad()
            pred = net(img)
            loss = crit(mask, pred)
            loss.backward(); opt.step()
            tot += loss.item()
        ep_loss = tot / len(dl)
        saved = ep_loss < best
        if saved:
            best, no_optim = ep_loss, 0
            torch.save(net.state_dict(), ckpt)
        else:
            no_optim += 1
        w.writerow([ep, f"{ep_loss:.5f}", f"{lr:.2e}", int(time.time() - tic), int(saved)]); log.flush()
        print(f"epoch {ep:3d}  loss {ep_loss:.4f}  lr {lr:.1e}  {int(time.time()-tic)}s{'  *' if saved else ''}", flush=True)
        if no_optim > 6:
            print(f"early stop at epoch {ep}"); break
        if no_optim > 3:
            if lr < 5e-7:
                break
            net.load_state_dict(torch.load(ckpt, map_location=dev))
            lr = lr / 5.0
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"  lr -> {lr:.1e} (reloaded best)")
    log.close()
    print("done", name, "best train loss", round(best, 5))


if __name__ == "__main__":
    main()
