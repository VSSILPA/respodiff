#!/bin/bash

# python  train_respodiff_safe.py \
# --seed 0 \
# --output_dir 'exps/exps_safe/sexual/0' \
# --concept "a gorgeous person" "sexual" \
# --neutral_concept "a gorgeous person" \
# --valid_prompt "a painting of a sensual lily trending on artstation in the style of greg rutkowski" \
# --max_train_steps 5000 \
# --target_guidance_scale 12 \
# --log_every_steps 500 \
# --train_batch_size 1 \
# --s_learning_rate 1e-4 \
# --c_learning_rate 1e-2 


python train_respodiff_safe.py \
--seed 0 \
--output_dir 'exps/exps_safe/violence/0' \
--concept "a scene" "violence" \
--neutral_concept "a scene" \
--valid_prompt "donald trump hitting people with pudding, pudding stained clothes, golden hour, boardwalk" \
--max_train_steps 5000 \
--target_guidance_scale 12 \
--log_every_steps 500 \
--train_batch_size 1 \
--s_learning_rate 1e-4 \
--c_learning_rate 1e-2 
