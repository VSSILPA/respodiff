#!/bin/bash

python safety_classify.py \
    --checkpoint "exps/exps_safe" \
    --root_dir "exps/exps_safe" \
    --fp16 \
    --num_gpus 2 \
    --concept sexual violence \
    --task "safe" 

python fid_clip_coco.py \
    --checkpoint "exps/exps_safe" \
    --root_dir "exps/exps_safe" \
    --fp16 \
    --num_gpus 2 \
    --concept sexual violence \
    --coco_src_img_dir "data/coco30k" \
    --task "safe" 

