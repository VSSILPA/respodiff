#!/bin/bash

python CLIP_classify.py \
--checkpoint "exps/exps_gender" \
--num_test_samples 15 \
--root_dir "exps/exps_gender/fair_eval/" \
--fp16 \
--num_gpus 3 \
--concept "a woman" "a man" \
--template_key "0" \
--clip_attributes "a woman" "a man"

python fid_clip_coco.py \
--checkpoint "exps/exps_gender" \
--root_dir "exps/exps_gender/fair_eval/" \
--fp16 \
--num_gpus 2 \
--concept "a woman" "a man" \
--coco_src_img_dir "data/coco30k" \
--task gender 

