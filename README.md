# M1: CBD-Net for oil-spill segmentation in Sentinel-1 SAR

Reproduction of **CBD-Net** (Q. Zhu *et al.*, "Oil spill contextual and boundary-supervised detection network based on marine SAR images," *IEEE Transactions on Geoscience and Remote Sensing*) using the **official code, unchanged** (byte-identical to github.com/CUG-URS/CBDNet-main, commit `ba3aa2e`, linked as a submodule), plus:
- a **leak-free** re-evaluation on a group-disjoint SOS split,
- **cross-sea tests** on Peru (new sea, labelled look-alikes) and MDOD (false alerts per look-alike type),
- every metric pooled over all test pixels, with bootstrap confidence intervals.

Method 1 of a three-method study (M2: Improved DeepLabV3+ → [sar-oil-spill-m2-deeplabv3plus](https://github.com/Vighnesh-SV/sar-oil-spill-m2-deeplabv3plus)).

## Results (TTA, oil class)

| Test | Trained on | Accuracy | Precision | Recall | **F1** | **IoU** | mIoU | ECE | MSE | R² |
|---|---|---|---|---|---|---|---|---|---|---|
| SOS official test (839) | official split (leaky) | 91.24 | 84.43 | 91.69 | **87.91** | **78.43** | 82.79 | 0.028 | 0.064 | 0.718 |
| SOS leak-free test (835) | leak-free split | 92.16 | 86.59 | 87.01 | **86.80** | **76.68** | 83.06 | 0.026 | 0.058 | 0.720 |
| Peru, all 2,013 patches | official split | 62.78 | 14.00 | 96.82 | 24.46 | 13.94 | 37.16 | 0.265 | 0.292 | −4.000 |
| Peru, all 2,013 patches | leak-free split | 71.56 | 17.41 | 95.29 | 29.44 | 17.26 | 43.51 | 0.186 | 0.220 | −2.770 |
| Published (independent re-run reported with FCS-Net) | official split | — | — | — | 87.81 | — | 82.94 | — | — | — |

MDOD (400 boxes per class): oil boxes flagged 64.0 % / 55.5 % (leaky / leak-free model), look-alike boxes 87–100 %; AUC oil vs look-alikes 0.282 / 0.335 (below 0.5 = look-alikes score higher than oil).

**Reading:** the reproduction is within 0.2 pp of the published numbers. Removing the leak costs about 1.7 pp oil IoU. On a new sea the model calls 86–90 % of look-alike pixels oil (precision 14–17 %), and its probabilities are over-confident (negative R²): the failure is **domain shift**, not overfitting (train − test oil IoU gap +0.1 / +1.3 pp).

All tables: [`docs/M1_CBDNet_All_Results.docx`](docs/M1_CBDNet_All_Results.docx); full report with figures and console output: [`M1_CBDNet/report/`](M1_CBDNet/report/); dataset checks and critique: [`docs/Datasets_Checks_Critique_QA.docx`](docs/Datasets_Checks_Critique_QA.docx).

## Repository structure

```
M1_CBDNet/            training script, one-file runner, report builders, results/, logs/, report/
common/               data loading (SOS), model registry, evaluation on SOS / Peru / MDOD (shared by M1–M3)
third_party/CBDNet-main   official CBD-Net code (git submodule, unchanged)
leakage_check/        proof of the SOS train/test leak, the scripts, and the leak-free split CSVs
runs/<run>/           saved metrics, per-tile/per-patch/per-box CSVs, training logs, figures (no weights)
results/summary.csv   one row per run, test set and mode
docs/                 all-results tables, analysis (overfitting, CIs, metrics), dataset checks and critique
data/                 put the datasets here (see data/README.md; not included)
```

## Setup (Windows, NVIDIA GPU)

```powershell
git clone --recurse-submodules https://github.com/Vighnesh-SV/sar-oil-spill-m1-cbdnet
cd sar-oil-spill-m1-cbdnet
python -m venv .venv          # Python 3.12
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```
If you cloned without `--recurse-submodules`, run `git submodule update --init` (this fetches the official CBD-Net code into `third_party/CBDNet-main`).

Then put the datasets in `data/` as described in [`data/README.md`](data/README.md). Pretrained ImageNet weights download automatically into `.cache/` on first use.

**Trained weights are not included** (about 100–125 MB each, and derived from academic-use-only data). Either train them with the runner below, or copy existing `best.pth` files into `runs/<run name>/`.

## Run

**From an editor (Antigravity / VS Code):** open the repository folder (`.vscode/settings.json` selects `.venv`), open `M1_CBDNet/run_m1.py`, set `STEP` and `SPLIT` at the top, press Run.

| STEP | Does |
|---|---|
| `check` (default) | lists what exists and prints the commands; nothing runs |
| `train` | trains CBD-Net on SOS Sentinel-1 (about 1 h 45 min on an RTX 3050 4 GB); refuses to overwrite a trained model unless `OVERWRITE = True` |
| `eval_sos`, `eval_peru`, `eval_mdod`, `all_tests` | tests the trained model |
| `report` | rebuilds `M1_CBDNet/results/` and the Word report from saved outputs |

`SPLIT = "leaky"` uses the official split, `"leakfree"` the group-disjoint split.

**From a terminal** (repository root):
```powershell
.venv\Scripts\python M1_CBDNet\train_m1.py                                   # official split
.venv\Scripts\python common\evaluate.py --run runs\M1_cbdnet_sentinel_dicebce_s0 --arch cbdnet
.venv\Scripts\python M1_CBDNet\train_m1.py --split leakage_check\splits\sos_sentinel_group_s0.csv
.venv\Scripts\python common\evaluate.py --run runs\M1_cbdnet_sentinel_dicebce_s0_groupsplit --arch cbdnet --split leakage_check\splits\sos_sentinel_group_s0.csv
.venv\Scripts\python common\evaluate_peru.py --run runs\<run> --arch cbdnet
.venv\Scripts\python common\evaluate_mdod.py --run runs\<run> --arch cbdnet
```

## Integrity checks and known limitations

Checked (details in `docs/Datasets_Checks_Critique_QA.docx`):
- **The official SOS split leaks.** Test tiles 20001–20839 are augmented copies of training areas (5 copies of 444 areas); 100 % of test tiles have copies in training. Proof and scripts: [`leakage_check/REPORT.md`](leakage_check/REPORT.md). Every model is therefore trained and tested twice: on the official (leaky) split, for comparison with the paper, and on a **group-disjoint (leak-free) split** that shares no sea area between partitions (verified: 0 shared areas in all partition pairs).
- Every SOS image has its mask (3,354/3,354 train, 839/839 test); no ID is in both train and test.
- SOS and Peru are greyscale stored as 3 identical channels and have similar brightness (mean 96 vs 88); MDOD is brighter (mean 135), about 10× coarser and has only boxes, so it is a **supplementary** look-alike stress test.
- The official evaluation script `t1p3-Iou.py` reports precision/recall/F1 for the **background** class (`labels=[255, 0]`). We report oil-class metrics as the main numbers and the script value separately.
- Test-time augmentation (8 flips/rotations) is inverted correctly; the decision threshold is fixed at 0.5; no test data is used for model selection.

Limitations:
- One training seed per model; SOS differences under about 1 pp are within noise (bootstrap CIs in the docs).
- The leak-free models were trained on train + val + calib (3,358 tiles), so no SOS data is held out for calibration. Any calibration or alert-threshold method must be fitted on Peru train (1,594 patches, 33 scenes) and reported only on Peru test (338 patches, 5 scenes).
- Peru classes 0/3/4 and the MDOD class IDs are inferred, not documented by the dataset authors (they affect only the per-class tables).

## Licence

Our code: MIT (see `LICENSE`). The official CBD-Net code is linked, not copied, and stays under its authors' rights. The datasets are not included and keep their own licences.
