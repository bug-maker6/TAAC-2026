#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}:${PYTHONPATH}"

export USE_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1100000 \
    --num_workers 8 \
    --save_epoch_ckpt 1 \
    --use_bf16 \
    "$@"
