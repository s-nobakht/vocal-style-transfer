#!/bin/bash

python -BW ignore wav2mel.py \
    --seed 2021 \
    --patience -1 \
    --root "/scratch-deleted-2021-mar-20/nobakht/style_trans/data" \
    --batch_size 128 \
    --num_workers 14 \
    --num_sanity_val_steps 2 \
    --min_steps 600000 \
    --max_steps 1000000 \
    --val_check_interval 0.0 \
    --gpus 4 \
    --auto_select_gpus true \
    --accelerator ddp \
    --benchmark true \
    --amp_backend native
