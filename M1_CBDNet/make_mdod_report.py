"""Build M1_CBDNet/results/mdod_external/ from the MDOD evaluations of both M1 models.

usage (from experiments/):  python M1_CBDNet/make_mdod_report.py
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = [("Official-split model", "M1_cbdnet_sentinel_dicebce_s0", "official"),
        ("Leak-free model", "M1_cbdnet_sentinel_dicebce_s0_groupsplit", "groupsplit")]
CLASSES = ["oil spill", "biogenic slick", "low wind", "rain cell", "sea ice"]
LOOKALIKES = CLASSES[1:]
INK, INK2, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def alert_chart(M, path):
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    for i, ((name, _, tag), col) in enumerate(zip(RUNS, [INK2, SERIES])):
        v = [100 * M[tag]["per_class"][c]["alert_rate@0.02"] for c in CLASSES]
        bars = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=col, label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, ["oil spill\n(should alert)"] + LOOKALIKES, fontsize=8.5, color=INK)
    ax.set_ylabel("boxes flagged as oil (%)", fontsize=9, color=INK2)
    ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    out = os.path.join(HERE, "results", "mdod_external"); os.makedirs(out, exist_ok=True)
    M = {}
    for _, run, tag in RUNS:
        d = os.path.join(EXP, "runs", run, "mdod")
        M[tag] = json.load(open(os.path.join(d, "metrics_mdod.json")))
        shutil.copy(os.path.join(d, "metrics_mdod.json"), os.path.join(out, f"metrics_{tag}.json"))
        shutil.copy(os.path.join(d, "qualitative_mdod.png"), os.path.join(out, f"qualitative_{tag}.png"))
    alert_chart(M, os.path.join(out, "alert_rates.png"))
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    s = M["official"]["settings"]
    L = ["# M1 CBD-Net: external test on MDOD (look-alike false alerts)", "",
         "**Data:** MDOD [N05], the OSF release: full Sentinel-1 scenes with **bounding boxes** in 5 classes "
         "(oil spill, biogenic slick, low wind, rain cell, sea ice). It has no pixel masks, so IoU and F1 cannot be "
         "computed. Instead we measure, for each box, how much of it the frozen model marks as oil.", "",
         f"**Protocol:** {s['per_class']} boxes per class, sampled with seed {s['seed']}; boxes smaller than "
         f"{s['min_side']} px are skipped. Each box is enlarged to at least 256 × 256 for context and predicted "
         "in 256 × 256 tiles, single pass (no TTA). The white land mask is excluded. "
         "*Coverage* = share of box pixels with p(oil) > 0.5. A box **alerts** if its coverage is at least 2 %.", "",
         "![](alert_rates.png)", "",
         "## Per class", "",
         "| Box class | Alert rate @2 %: official | Alert rate @2 %: leak-free | Alert rate @20 %: official | Alert rate @20 %: leak-free | Mean p(oil): official | Mean p(oil): leak-free |",
         "|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        o, g = M["official"]["per_class"][c], M["groupsplit"]["per_class"][c]
        name = f"**{c}** (should alert)" if c == "oil spill" else c
        L.append(f"| {name} | {pct(o['alert_rate@0.02'])} | {pct(g['alert_rate@0.02'])} | "
                 f"{pct(o['alert_rate@0.2'])} | {pct(g['alert_rate@0.2'])} | "
                 f"{o['mean_box_score']:.2f} | {g['mean_box_score']:.2f} |")
    L += ["", "## Can the model rank oil above look-alikes? (AUC, oil boxes vs each look-alike class)", "",
          "AUC is the chance that a random oil box gets a higher score than a random look-alike box. "
          "0.5 = coin flip, 1.0 = perfect, **below 0.5 = look-alikes score higher than oil**.", "",
          "| Oil vs | Official (box mean) | Leak-free (box mean) | Official (box peak) | Leak-free (box peak) |",
          "|---|---|---|---|---|"]
    for c in LOOKALIKES + ["all look-alikes"]:
        L.append(f"| {c} | {M['official']['auc_oil_vs'][c]:.3f} | {M['groupsplit']['auc_oil_vs'][c]:.3f} | "
                 f"{M['official']['auc_oil_vs_peak'][c]:.3f} | {M['groupsplit']['auc_oil_vs_peak'][c]:.3f} |")
    L += ["", "## Findings",
          "1. **Look-alikes trigger more alerts than real oil.** 87–100 % of look-alike boxes are flagged, but only "
          "56–64 % of oil boxes. Low wind is the worst: almost every box is flagged, usually over half its area.",
          "2. **AUC is below 0.5 for every look-alike class (0.16–0.40).** M1 has learned 'dark and smooth = oil', "
          "not the shape of a slick, so the most oil-like signal comes from large dark areas.",
          "3. **The leak-free model is slightly less trigger-happy** (lower alert rates, AUC 0.34 vs 0.28), but it also "
          "flags less oil. Same pattern as on Peru; single seed.",
          "4. **Caveat: scale.** MDOD scenes are about 10× coarser than SOS tiles, so many MDOD oil spills are thin "
          "streaks only a few pixels wide that the model misses. Part of the low oil alert rate is a resolution "
          "mismatch, not only a model weakness. The look-alike false alerts are real at any scale.",
          "5. **Together with Peru** (86–90 % of look-alike pixels called oil), this is the evidence for C4 "
          "(wind- and look-alike-aware alerting): a detector trained on SOS cannot reject look-alikes by itself.", "",
          "## Qualitative (`qualitative_official.png`, `qualitative_groupsplit.png`)",
          "- Low-wind areas and the dark sides of rain cells and sea-ice edges light up as oil across most of the box.",
          "- Biogenic slicks: the filaments are traced as oil.",
          "- Thin oil streaks are often missed entirely (coverage 2 %); wider slicks are found.", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
