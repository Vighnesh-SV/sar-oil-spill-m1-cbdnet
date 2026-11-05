"""M1 CBD-Net: one file to run every step. Open it in the editor and press Run (no arguments needed).

1. Choose STEP and SPLIT below.
2. Run this file with the project interpreter: .venv\\Scripts\\python.exe

It only calls the existing scripts (train_m1.py, common/evaluate*.py, make_*report.py) with the exact
arguments used for the reported results, so the method code is unchanged.
"""
import os
import subprocess
import sys

# ------------------------------- settings -------------------------------
STEP = "check"
#   "check"      list what exists and print the commands (no GPU, nothing written)
#   "train"      train on SOS Sentinel-1                     (about 1 h 45 min)
#   "eval_sos"   test on the SOS test set of the chosen split (about 5 min)
#   "eval_peru"  test on Peru (external, all 2,013 patches)
#   "eval_mdod"  test on MDOD (external, 400 boxes per class)
#   "report"     rebuild results/ folders and the Word report from saved outputs (no GPU)
#   "all_tests"  eval_sos + eval_peru + eval_mdod
SPLIT = "leaky"        # "leaky" = official SOS split, "leakfree" = group-disjoint split
OVERWRITE = False      # training refuses to replace an existing trained model unless True
# ------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SPLIT_CSV = os.path.join("leakage_check", "splits", "sos_sentinel_group_s0.csv")
RUN = os.path.join("runs", "M1_cbdnet_sentinel_dicebce_s0" + ("_groupsplit" if SPLIT == "leakfree" else ""))
split_args = ["--split", SPLIT_CSV] if SPLIT == "leakfree" else []

COMMANDS = {
    "train": [["M1_CBDNet/train_m1.py", *split_args]],
    "eval_sos": [["common/evaluate.py", "--run", RUN, "--arch", "cbdnet", *split_args]],
    "eval_peru": [["common/evaluate_peru.py", "--run", RUN, "--arch", "cbdnet"]],
    "eval_mdod": [["common/evaluate_mdod.py", "--run", RUN, "--arch", "cbdnet"]],
    "report": [
        ["M1_CBDNet/make_report.py"],
        ["M1_CBDNet/make_report.py", "--run", "runs/M1_cbdnet_sentinel_dicebce_s0_groupsplit", "--split", "groupsplit",
         "--tag", "sentinel_group_split"],
        ["M1_CBDNet/make_peru_report.py"],
        ["M1_CBDNet/make_mdod_report.py"],
        ["M1_CBDNet/report/make_terminal_pics.py"],
        ["M1_CBDNet/report/build_m1_report.py"],
    ],
}
COMMANDS["all_tests"] = COMMANDS["eval_sos"] + COMMANDS["eval_peru"] + COMMANDS["eval_mdod"]


def run(cmd):
    full = [sys.executable, "-W", "ignore", "-u", *cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(full, cwd=EXP, check=True)


def check():
    print(f"Python: {sys.executable}")
    if sys.prefix == sys.base_prefix:
        print("  WARNING: not running inside a virtual environment. Select .venv\\Scripts\\python.exe (see README).")
    for name in ("M1_cbdnet_sentinel_dicebce_s0", "M1_cbdnet_sentinel_dicebce_s0_groupsplit"):
        d = os.path.join(EXP, "runs", name)
        print(f"{name}: model {'OK' if os.path.exists(os.path.join(d, 'best.pth')) else 'MISSING'}, "
              f"Peru {'done' if os.path.isdir(os.path.join(d, 'peru')) else '-'}, MDOD {'done' if os.path.isdir(os.path.join(d, 'mdod')) else '-'}")
    print(f"\nCommands for SPLIT = {SPLIT!r} (run from the repository root):")
    for step, cmds in COMMANDS.items():
        for c in cmds:
            print(f"  [{step}] python {' '.join(c)}")


if __name__ == "__main__":
    if STEP == "check":
        check()
    elif STEP not in COMMANDS:
        sys.exit(f"Unknown STEP {STEP!r}; choose one of: check, {', '.join(COMMANDS)}")
    else:
        if STEP == "train" and os.path.exists(os.path.join(EXP, RUN, "best.pth")) and not OVERWRITE:
            sys.exit(f"{RUN} already has a trained model (best.pth). Set OVERWRITE = True to retrain and replace it.")
        for c in COMMANDS[STEP]:
            run(c)
        print("\nDone.")
