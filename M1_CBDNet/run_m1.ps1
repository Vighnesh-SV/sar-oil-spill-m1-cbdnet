# M1 CBD-Net: full reproduction in one command (train -> evaluate -> report).
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File M1_CBDNet\run_m1.ps1
# Takes about 1 h 45 min on an RTX 3050 laptop GPU (4 GB).
param([string]$Sensor = "sentinel", [string]$Loss = "dicebce", [int]$Seed = 0)

$ErrorActionPreference = "Stop"
$py  = ".venv\Scripts\python.exe"
$run = "runs\M1_cbdnet_${Sensor}_${Loss}_s${Seed}"
$log = "M1_CBDNet\logs\train_eval_${Sensor}_${Loss}_s${Seed}.log"

& $py -W ignore -u M1_CBDNet\train_m1.py --sensor $Sensor --loss $Loss --seed $Seed *>&1 | Tee-Object -FilePath $log
& $py -W ignore -u common\evaluate.py --run $run --sensor $Sensor --arch cbdnet *>&1 | Tee-Object -FilePath $log -Append
& $py M1_CBDNet\make_report.py --run $run --sensor $Sensor --tag "${Sensor}_official_split"
