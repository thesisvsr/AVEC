#!/usr/bin/env bash
# Helper script to ensure venv is activated before launching BSRD training.
# Usage examples:
#  ./scripts/run_bsrd_train.sh resume
#  ./scripts/run_bsrd_train.sh fresh
#  ./scripts/run_bsrd_train.sh custom -i checkpoints_epoch_50_step_1992.ckpt --steps_per_epoch 120

set -euo pipefail

# Root of repo (directory of this script -> up one if needed)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -d .venv ]]; then
  echo "[ERROR] .venv directory not found. Create it first (e.g. python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)." >&2
  exit 1
fi

# Activate virtual environment explicitly BEFORE any python command
# (Per user requirement)
source .venv/bin/activate

# Common env overrides (edit as needed or export externally)
export BSRD_BATCH=${BSRD_BATCH:-4}
export BSRD_ENTROPY_LAMBDA=${BSRD_ENTROPY_LAMBDA:-0.005}
export BSRD_LM_GATE_EPOCH=${BSRD_LM_GATE_EPOCH:-5}
export BSRD_LM_GATE_BLANK=${BSRD_LM_GATE_BLANK:-0.92}
export BSRD_LM_RAMP_EPOCHS=${BSRD_LM_RAMP_EPOCHS:-6}
# export BSRD_KENLM_PATH=${BSRD_KENLM_PATH:-export/kenlm_char.binary}
export PROGRESS_TQDM_ONLY=${PROGRESS_TQDM_ONLY:-0}  # default show epoch header + bar; set to 1 for pure bar only
export FORCE_PROGRESS_BAR=${FORCE_PROGRESS_BAR:-1}   # ensure bar visible even if prints occur
# Retain last K checkpoints in addition to best (config logic reads RETAIN_LAST_K or BSRD_RETAIN_LAST_CKPTS)
export RETAIN_LAST_K=${RETAIN_LAST_K:-5}

CFG="configs/BSRD/VO/EffConfInterCTC.py"
# Allow high worker count (user requested >40). WARNING: ensure enough RAM.
NUM_WORKERS=${NUM_WORKERS:-48}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-120}
EVAL_STEPS=${EVAL_STEPS:-40}
SAVE_PERIOD_EPOCH=${SAVE_PERIOD_EPOCH:-1}
EVAL_PERIOD_EPOCH=${EVAL_PERIOD_EPOCH:-1}
STEP_LOG_PERIOD=${STEP_LOG_PERIOD:-30}

CMD_MODE="fresh"
if [[ $# -gt 0 ]]; then
  CMD_MODE=$1; shift || true
fi

EXTRA_ARGS=("$@")

# Auto-detect latest valid checkpoint if resuming
latest_ckpt=""
if [[ "$CMD_MODE" == "resume" || "$CMD_MODE" == "auto" ]]; then
  python3 - <<'PY' > .latest_ckpt_tmp 2>/dev/null
import glob,re,torch,os,sys
pat=re.compile(r'epoch_(\d+)_step_(\d+)')
ckpts=glob.glob('callbacks/BSRD/VO/EffConfInterCTC/checkpoints_epoch_*_step_*.ckpt')
meta=[]
for p in ckpts:
 m=pat.search(p)
 if not m: continue
 try:
  torch.load(p, map_location='cpu')
  meta.append((int(m.group(1)), int(m.group(2)), p))
 except Exception:
  pass
meta.sort()
if meta:
 print(meta[-1][2])
PY
  if [[ -s .latest_ckpt_tmp ]]; then
    latest_ckpt=$(cat .latest_ckpt_tmp)
    echo "[INFO] Resuming from $latest_ckpt"
  else
    echo "[WARN] No valid checkpoint found; starting fresh." >&2
  fi
  rm -f .latest_ckpt_tmp
fi

set -x
if [[ -n "$latest_ckpt" ]]; then
  python main.py -c "$CFG" -m training -i "$(basename "$latest_ckpt")" -j "$NUM_WORKERS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" --eval_steps "$EVAL_STEPS" \
    --eval_period_epoch "$EVAL_PERIOD_EPOCH" --saving_period_epoch "$SAVE_PERIOD_EPOCH" \
    --step_log_period "$STEP_LOG_PERIOD" "${EXTRA_ARGS[@]}"
else
  python main.py -c "$CFG" -m training -j "$NUM_WORKERS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" --eval_steps "$EVAL_STEPS" \
    --eval_period_epoch "$EVAL_PERIOD_EPOCH" --saving_period_epoch "$SAVE_PERIOD_EPOCH" \
    --step_log_period "$STEP_LOG_PERIOD" "${EXTRA_ARGS[@]}"
fi
set +x

