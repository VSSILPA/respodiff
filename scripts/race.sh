#!/bin/bash

for concept in "a Asian-race person" "a black-race person" "a white-race person"
do
  python train_respodiff_fair.py \
    --seed 0 \
    --valid_prompt "a photo of a doctor" \
    --concept "$concept" \
    --output_dir "exps/exps_race/$concept/0" \
    --max_train_steps 5000 \
    --log_every_steps 500 \
    --train_batch_size 1 \
    --s_learning_rate 1e-2 \
    --c_learning_rate 1e-1 \
    --lambda_align 0.3
done
