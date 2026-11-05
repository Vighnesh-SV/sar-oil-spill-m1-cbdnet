"""Build the presentable M1 results folder from a finished run.

Reads   runs/<run>/{train_log.csv, metrics_<sensor>.json, per_tile_<sensor>.csv, qualitative_<sensor>.png}
Writes  M1_CBDNet/results/<tag>/
          results.md            metric table vs published values + notes
          training_curve.png    training loss per epoch (learning-rate drops marked)
          per_tile_iou.png      distribution of per-tile oil IoU on the test set
          qualitative.png       SAR / ground truth / TP-FP-FN overlays (median + worst tiles)
          metrics.json, per_tile.csv, train_log.csv   (copies of the raw numbers)

usage (from experiments/):  python M1_CBDNet/make_report.py --run runs/M1_cbdnet_sentinel_dicebce_s0
  group split (C1):  python M1_CBDNet/make_report.py --run runs/M1_cbdnet_sentinel_dicebce_s0_groupsplit \n                         --split groupsplit --tag sentinel_group_split
"""
import argparse
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

# reference-palette roles (light): one series -> slot 1; text and grid stay neutral
SERIES, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

# Published targets for SOS (review Table 3 / Section 7.2)
PUBLISHED = {"sentinel": {"miou": 82.94, "f1_oil": 87.81, "src": "independent re-run of CBD-Net in [P29]"},
             "palsar": {"miou": 83.96, "f1_oil": 85.07, "src": "independent re-run of CBD-Net in [P29]"}}


def training_curve(log_csv, out_png):
    rows = list(csv.DictReader(open(log_csv)))
    ep = np.array([int(r["epoch"]) for r in rows]); loss = np.array([float(r["train_loss"]) for r in rows])
    lr = [float(r["lr"]) for r in rows]
    best = int(ep[np.argmin(loss)])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ep, loss, color=SERIES, lw=2)
    ax.scatter([best], [loss.min()], s=40, color=SERIES, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"best {loss.min():.3f} (epoch {best})", (best, loss.min()), xytext=(-8, 14),
                textcoords="offset points", ha="right", color=INK, fontsize=8)
    drops = [i for i in range(1, len(lr)) if lr[i] < lr[i - 1]]
    step, row = (loss.max() - loss.min()) * 0.07, 0
    for n, i in enumerate(drops):
        row = row + 1 if n and ep[i] - ep[drops[n - 1]] < 6 else 0   # stagger labels of close drops
        ax.axvline(ep[i] - 0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.text(ep[i] - 0.5, loss.max() - row * step, f" lr → {lr[i]:.1e}", color=INK2, fontsize=7, va="top")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (BCE + Dice)")
    ax.set_title("M1 CBD-Net: training loss on SOS Sentinel-1", loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return best, float(loss.min()), int(ep[-1])


def per_tile_hist(per_tile_csv, out_png):
    rows = [r for r in csv.DictReader(open(per_tile_csv)) if int(r["gt_oil_px"]) > 0]
    iou = np.array([float(r["iou_oil"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(iou, bins=np.linspace(0, 1, 21), color=SERIES, edgecolor="white", linewidth=2)
    med = float(np.median(iou))
    ax.axvline(med, color=INK, lw=1)
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.2f}", color=INK, fontsize=8, va="top")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("oil IoU of one test tile"); ax.set_ylabel("number of tiles")
    ax.set_title(f"Per-tile oil IoU ({len(iou)} test tiles that contain oil)", loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return med, float((iou < 0.3).mean())


def write_group_report(a, m, out, best, best_loss, last, med, frac_bad):
    """C1 report: the same M1 setup trained on the group-disjoint split vs the official split."""
    base = json.load(open(os.path.join(EXP, a.baseline, f"metrics_{a.sensor}.json")))["tta"]
    t = m["tta"]
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    rows = [("**Oil IoU**", "iou_oil"), ("**Oil F1**", "f1_oil"), ("Precision (oil)", "precision_oil"),
            ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("IoU (background)", "iou_bg"),
            ("Accuracy", "accuracy"), ("Boundary F1 (2 px)", "boundary_f1"), ("ECE (×100)", "ece"),
            ("Oil share of test pixels", "oil_pixel_fraction")]
    L = [f"# M1 CBD-Net: group-disjoint split (C1 leakage test), SOS {a.sensor.capitalize()}", "",
         "**Question:** how much does the official SOS split flatter the scores? Every official test tile has "
         "augmented copies of the same sea area in training (`leakage_check/REPORT.md`).", "",
         "**Setup:** identical model, loss, schedule and seed to the official run. Only the split changes.",
         f"- Train: {3358} tiles (partitions train + val + calib), about the official 3,354, so training size is not a factor.",
         f"- Test: {m['n_tiles']} tiles from sea areas never seen in training (0 shared tiles, 0 shared groups).",
         f"- Training stopped at epoch {last}; best training loss {best_loss:.4f} at epoch {best}.", "",
         "## Results (%, with TTA)", "",
         "| Metric | Official split (leaky) | **Group-disjoint** | Difference |", "|---|---|---|---|"]
    for name, k in rows:
        L.append(f"| {name} | {pct(base[k])} | **{pct(t[k])}** | {100 * (t[k] - base[k]):+.2f} |")
    L += ["", "## Interpretation",
          f"- **The leakage effect is small:** oil IoU {100 * (t['iou_oil'] - base['iou_oil']):+.2f} pp and oil F1 "
          f"{100 * (t['f1_oil'] - base['f1_oil']):+.2f} pp. On unseen areas the model finds slightly less oil "
          f"(recall {100 * (t['recall_oil'] - base['recall_oil']):+.2f} pp).",
          "- **mIoU hides the drop.** The group test set has less oil "
          f"({pct(t['oil_pixel_fraction'])} % vs {pct(base['oil_pixel_fraction'])} % of pixels), so background IoU rises "
          "and offsets the loss in oil IoU. Oil IoU and F1 are the fair comparison here.",
          "- Calibration (ECE) is unchanged.", "",
          "## Caveats",
          "1. **One seed.** Differences of 1–2 pp are close to run-to-run variation. Repeat with 3 seeds before claiming a number.",
          "2. **Groups are sea areas (crops), not the 21 original scenes.** Scene-level leakage remains possible, "
          "so this is a lower bound. The Peru dataset's scene split can test it.",
          "3. The two test sets differ (different tiles), so part of any difference comes from test composition.", "",
          "## Figures",
          "- `training_curve.png`, `per_tile_iou.png` (median "
          f"{med:.2f}; {100 * frac_bad:.0f}% of oil tiles below 0.3), `qualitative.png`: same format as the official run.", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/M1_cbdnet_sentinel_dicebce_s0")
    ap.add_argument("--sensor", default="sentinel")
    ap.add_argument("--tag", default="sentinel_official_split")
    ap.add_argument("--split", default="", help="'groupsplit' -> compare against the official-split run")
    ap.add_argument("--baseline", default="runs/M1_cbdnet_sentinel_dicebce_s0")
    a = ap.parse_args()
    run = os.path.join(EXP, a.run); out = os.path.join(HERE, "results", a.tag); os.makedirs(out, exist_ok=True)

    sfx = a.sensor + (f"_{a.split}" if a.split else "")
    m = json.load(open(os.path.join(run, f"metrics_{sfx}.json")))
    best, best_loss, last = training_curve(os.path.join(run, "train_log.csv"), os.path.join(out, "training_curve.png"))
    med, frac_bad = per_tile_hist(os.path.join(run, f"per_tile_{sfx}.csv"), os.path.join(out, "per_tile_iou.png"))
    shutil.copy(os.path.join(run, f"qualitative_{sfx}.png"), os.path.join(out, "qualitative.png"))
    shutil.copy(os.path.join(run, f"metrics_{sfx}.json"), os.path.join(out, "metrics.json"))
    shutil.copy(os.path.join(run, f"per_tile_{sfx}.csv"), os.path.join(out, "per_tile.csv"))
    shutil.copy(os.path.join(run, "train_log.csv"), os.path.join(out, "train_log.csv"))

    if a.split:
        return write_group_report(a, m, out, best, best_loss, last, med, frac_bad)
    t, p, pub = m["tta"], m["plain"], PUBLISHED[a.sensor]
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    rows = [("Accuracy", "accuracy"), ("Precision (oil)", "precision_oil"), ("Recall (oil)", "recall_oil"),
            ("**F1 (oil)**", "f1_oil"), ("IoU (oil)", "iou_oil"), ("IoU (background)", "iou_bg"),
            ("**mIoU**", "miou"), ("Boundary F1 (2 px)", "boundary_f1"),
            ("F1 as computed by the official script (background class)", "f1_bg_officialscript")]
    pubv = {"miou": pub["miou"], "f1_oil": pub["f1_oil"]}
    lines = [f"# M1 CBD-Net: results on SOS {a.sensor.capitalize()} (official split)", "",
             f"Run: `{a.run}`. Test tiles: {m['n_tiles']}. Parameters: {m['params_M']:.1f} M. "
             f"Training stopped at epoch {last}, best training loss {best_loss:.4f} at epoch {best}.", "",
             "## Quantitative results (%, pixels pooled over all test tiles)", "",
             "| Metric | Ours, with TTA | Ours, single pass | Published |", "|---|---|---|---|"]
    for name, k in rows:
        lines.append(f"| {name} | {pct(t[k])} | {pct(p[k])} | {pubv.get(k, '—')} |")
    lines += ["", "| Probability quality | With TTA | Single pass |", "|---|---|---|",
              f"| ROC-AUC | {t['roc_auc']:.4f} | {p['roc_auc']:.4f} |",
              f"| PR-AUC | {t['pr_auc']:.4f} | {p['pr_auc']:.4f} |",
              f"| ECE (15 bins) | {t['ece']:.4f} | {p['ece']:.4f} |",
              f"| Brier score | {t['brier']:.4f} | {p['brier']:.4f} |", "",
              f"Published values: {pub['src']}. TTA = the official 8-way rotation/flip test-time augmentation.",
              f"**Reproduction gap:** mIoU {100 * t['miou'] - pub['miou']:+.2f} pp, "
              f"F1 {100 * t['f1_oil'] - pub['f1_oil']:+.2f} pp (target: within ±2 pp).", "",
              "## Figures", "",
              "- `training_curve.png`: training loss per epoch; dashed lines mark learning-rate drops.",
              f"- `per_tile_iou.png`: per-tile oil IoU. Median {med:.2f}; {100 * frac_bad:.0f}% of oil tiles score below 0.3.",
              "- `qualitative.png`: four median and four worst test tiles (green = correct oil, red = false oil, blue = missed oil).", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
