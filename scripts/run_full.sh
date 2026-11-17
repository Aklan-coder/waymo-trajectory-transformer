#!/bin/bash
# Full pipeline. Run inside tmux so it survives a dropped SSH connection:
#   tmux new -s waymo      then      bash run_full.sh 1000      (or 100, 300 ...)
# Detach: Ctrl+B then D.   Re-attach later: tmux attach -t waymo
set -e
cd "$(dirname "$0")/.."
source ~/venv/bin/activate
source ~/google-cloud-sdk/path.bash.inc 2>/dev/null || true
N=${1:-1000}                       # number of training files (max 1000)
EPOCHS=${2:-10}                    # with much more data, 10 epochs is enough
DATA=data_${N}
RUNS=runs_${N}
echo "=== $(date) | $N training files, $EPOCHS epochs ==="
python fetch_convert.py --split validation --start 0 --end 5  --out $DATA/val
python fetch_convert.py --split validation --start 5 --end 10 --out $DATA/test
python fetch_convert.py --split training   --start 0 --end $N --out $DATA/train
echo "=== $(date) | data ready: $(ls $DATA/train | wc -l) train files ==="
python train.py --data $DATA --eval_cv
python train.py --data $DATA --K 1 --epochs $EPOCHS --out $RUNS 2>&1 | tee train_K1_${N}.log
python train.py --data $DATA --K 3 --epochs $EPOCHS --out $RUNS 2>&1 | tee train_K3_${N}.log
python analyze.py --data $DATA --runs $RUNS --split test
echo "=== $(date) | ALL DONE. Results in $RUNS/analysis_test ==="
