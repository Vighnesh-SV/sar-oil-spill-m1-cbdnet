"""Build M1_CBDNet/results/peru_external/ from the Peru evaluations of both M1 models.

usage (from experiments/):  python M1_CBDNet/make_peru_report.py
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = [("Official-split model", "M1_cbdnet_sentinel_dicebce_s0", "official"),
        ("Leak-free (group-split) model", "M1_cbdnet_sentinel_dicebce_s0_groupsplit", "groupsplit")]


def main():
    out = os.path.join(HERE, "results", "peru_external"); os.makedirs(out, exist_ok=True)
    M, S = {}, {}
    for _, run, tag in RUNS:
        d = os.path.join(EXP, "runs", run, "peru")
        M[tag] = json.load(open(os.path.join(d, "metrics_peru.json")))
        shutil.copy(os.path.join(d, "metrics_peru.json"), os.path.join(out, f"metrics_{tag}.json"))
        shutil.copy(os.path.join(d, "qualitative_peru.png"), os.path.join(out, f"qualitative_{tag}.png"))
        sos_file = "metrics_sentinel.json" if tag == "official" else "metrics_sentinel_groupsplit.json"
        S[tag] = json.load(open(os.path.join(EXP, "runs", run, sos_file)))["tta"]
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    L = ["# M1 CBD-Net: external test on Peru (new sea, never used for training)", "",
         "**Data:** Peru set [N22], corrected version: 2,013 Sentinel-1 patches (512 × 512) from 40 scenes, 2014–2024. "
         "Labels: oil, look-alike, ship, land, sea. Scoring is binary (oil vs everything else) with the same pooled "
         "metrics and 8-way TTA as the SOS test. Each patch is cut into four 256 × 256 tiles, the training size.", "",
         "## SOS (in-domain) vs Peru (new sea), all 2,013 patches", "",
         "| Metric (%) | Official model: SOS | **Official model: Peru** | Leak-free model: SOS | **Leak-free model: Peru** |",
         "|---|---|---|---|---|"]
    for name, k in [("**Oil F1**", "f1_oil"), ("**Oil IoU**", "iou_oil"), ("Precision (oil)", "precision_oil"),
                    ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("Accuracy", "accuracy")]:
        L.append(f"| {name} | {pct(S['official'][k])} | **{pct(M['official']['all'][k])}** | "
                 f"{pct(S['groupsplit'][k])} | **{pct(M['groupsplit']['all'][k])}** |")
    L.append(f"| ROC-AUC (×100) | {pct(S['official']['roc_auc'])} | **{pct(M['official']['all']['roc_auc'])}** | "
             f"{pct(S['groupsplit']['roc_auc'])} | **{pct(M['groupsplit']['all']['roc_auc'])}** |")
    L.append(f"| ECE (×100) | {pct(S['official']['ece'])} | **{pct(M['official']['all']['ece'])}** | "
             f"{pct(S['groupsplit']['ece'])} | **{pct(M['groupsplit']['all']['ece'])}** |")
    L += ["", "## What the model calls oil on Peru (share of each true class predicted as oil, %)", "",
          "| True class | Share of Peru pixels | Official model | Leak-free model |", "|---|---|---|---|"]
    for c in ["oil", "look-alike", "ship", "sea", "land"]:
        L.append(f"| {c} | {pct(M['official']['all']['class_pixel_share'][c])} | "
                 f"{pct(M['official']['all']['called_oil_rate_per_class'][c])} | "
                 f"{pct(M['groupsplit']['all']['called_oil_rate_per_class'][c])} |")
    o, g = M["official"]["official_test"], M["groupsplit"]["official_test"]
    L += ["", f"Peru's own 338-patch test subset (5 scenes): oil F1 {pct(o['f1_oil'])} / {pct(g['f1_oil'])}, "
          f"oil IoU {pct(o['iou_oil'])} / {pct(g['iou_oil'])} (official / leak-free model).", "",
          "## Findings",
          "1. **Severe drop under domain shift:** oil F1 falls from about 87 % on SOS to 24–29 % on Peru.",
          "2. **Recall stays high (95–97 %) but precision collapses (14–17 %).** The model calls almost every dark area "
          "oil: 86–90 % of look-alike pixels, 45–69 % of ship pixels and 28–38 % of open sea. "
          "About 6 of every 7 oil alarms are false.",
          "3. **The ranking is still good (ROC-AUC 0.92–0.93), but the 0.5 threshold and the probabilities are wrong**: "
          "ECE rises from about 0.03 to 0.19–0.26. The model is confidently wrong on the new sea. "
          "This is direct evidence for the calibration (C2) and alert-threshold (C3) contributions.",
          "4. The leak-free model transfers slightly better (F1 +5 pp, ECE lower). This is a single seed.",
          "5. Not a data bug: Peru and SOS have similar sea brightness (median 86 vs 89). Peru has lower contrast and "
          "darker oil, which is genuine domain shift.", "",
          "## Qualitative (`qualitative_official.png`, `qualitative_groupsplit.png`)",
          "- **Oil patches:** the slicks are found (green), but the model spreads oil into the surrounding dark sea and "
          "along the coastline (red).",
          "- **Look-alike patches** (whole tiles of low-wind or other dark water) are predicted as oil almost everywhere. "
          "SOS has no look-alike class, so the model never learned to reject them.", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
