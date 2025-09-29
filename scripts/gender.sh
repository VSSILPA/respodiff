#!/bin/bash

python train_respodiff_fair.py \
--seed 0 \
--valid_prompt "a photo of a doctor" \
--concept "a man" \
--output_dir "exps/exps_gender/a man/0" \
--max_train_steps 5000 \
--log_every_steps 500 \
--train_batch_size 1 \
--s_learning_rate 1e-4 \
--c_learning_rate 1e-2

python train_respodiff_fair.py \
--seed 0 \
--valid_prompt "a photo of a doctor" \
--concept "a woman" \
--output_dir "exps/exps_gender/a woman/0" \
--max_train_steps 5000 \
--log_every_steps 500 \
--train_batch_size 1 \
--s_learning_rate 1e-3 \
--c_learning_rate 1e-2