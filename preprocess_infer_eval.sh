#!/usr/bin/env bash
set -euo pipefail

# Prefer local src/ first
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# ADATA="/home/hackerman/Github/state/competition_support_set/hepg2.h5"
ADATA="/home/hackerman/Github/state/filtered_set/hepg2_filtered.h5"
#MODEL_DIR="/home/hackerman/Github/axonome-state/competition/2025-07-28T18:55:06.027430/"
MODEL_DIR="/home/hackerman/Downloads/hepg2_overfit/"
CHECKPOINT="step=40000.ckpt"
EVAL_DIR="./cell_eval_results_filt"


# ---- stems via Bash parameter expansion (no Python subprocess) ----
# hepg2.h5 -> hepg2
DATA_STEM="${ADATA##*/}"; DATA_STEM="${DATA_STEM%.*}"

# 2025-07-28T18:55:06.027430/ -> 2025-07-28T18:55:06.027430
MODEL_NAME="${MODEL_DIR%/}"; MODEL_NAME="${MODEL_NAME##*/}"
# -------------------------------------------------------------------

PREPRO_DIR="/home/hackerman/Github/state/competition_support_set/validation_data"
OUT_DIR_BASE="/home/hackerman/Github/state/competition"

mkdir -p "$PREPRO_DIR" "$OUT_DIR_BASE"

# Seeds to test; adjust as needed
#NUM_WORKERS=(1 2 4 8 12)
#for NUM_WORKER in "${NUM_WORKERS[@]}"; do

  SEED=42
  prepro=true  # or "false"
  printf "\n\n=== Running with seed %s ===\n" "$SEED"
  if [[ "$prepro" == "true" ]]; then
    echo "RUNNING WITH PREPROCESSING"
    PREPRO_PATH="${PREpro_DIR:-$PREPRO_DIR}/${DATA_STEM}_preprocessed2_s${SEED}.h5"
    python -m state tx preprocess_infer \
      --adata="$ADATA" \
      --output="$PREPRO_PATH" \
      --control_condition="non-targeting" \
      --pert_col="target_gene" \
      --seed="$SEED"
    NAME="${MODEL_NAME}_${DATA_STEM}_s${SEED}_with_replace"
    OUTPUT_PATH="${OUT_DIR_BASE}/${NAME}.h5ad"
    python -m state tx infer \
    --adata="$PREPRO_PATH" \
    --output="$OUTPUT_PATH" \
    --model_dir="$MODEL_DIR" \
    --checkpoint=$CHECKPOINT \
    --pert_col="target_gene" \
    --ctrl_pert="non-targeting"

    OUTPUT_DIR="${EVAL_DIR}/cell-eval-${NAME}"
    mkdir -p "$OUTPUT_DIR"

    python -m cell_eval run \
      --profile=vcc \
      -ar="$ADATA" \
      -ap="$OUTPUT_PATH" \
      --num-threads=12 \
      --outdir="$OUTPUT_DIR"
  else
    echo "RUNNING WITHOUT PREPROCESSING"
    PREPRO_PATH=$ADATA
    OUTPUT_PATH="${OUT_DIR_BASE}/${MODEL_NAME}_${DATA_STEM}_dataloader.h5ad"
    python -m state tx infer \
      --adata="$PREPRO_PATH" \
      --output="$OUTPUT_PATH" \
      --model_dir="$MODEL_DIR" \
      --checkpoint=$CHECKPOINT \
      --pert_col="target_gene" \
      --ctrl_pert="non-targeting"

    # OUTPUT_DIR="./cell_eval_results_bash/cell-eval-${MODEL_NAME}_${DATA_STEM}_s${SEED}"
    OUTPUT_DIR="${EVAL_DIR}/cell-eval-${MODEL_NAME}_${DATA_STEM}"
    mkdir -p "$OUTPUT_DIR"

    python -m cell_eval run \
      --profile=vcc \
      -ar="$ADATA" \
      -ap="$OUTPUT_PATH" \
      --num-threads=12 \
      --outdir="$OUTPUT_DIR"
    # done
  fi
#done
