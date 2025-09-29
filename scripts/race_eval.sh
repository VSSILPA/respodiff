#!/bin/bash

python CLIP_classify.py \
--checkpoint "exps/exps_race" \
--num_test_samples 10 \
--root_dir "exps/exps_race/fair_eval/" \
--fp16 \
--num_gpus 3 \
--concept "a black-race person" "a white-race person" "a Asian-race person" \
--template_key "0" \
--clip_attributes "a black-race person" "a white-race person" "a Asian-race person"


python fid_clip_coco.py \
--checkpoint "exps/exps_race" \
--root_dir "exps/exps_race/fair_eval/" \
--fp16 \
--num_gpus 3 \
--concept "a black-race person" "a white-race person" "a Asian-race person" \
--coco_src_img_dir "data/coco30k" \
--task race


