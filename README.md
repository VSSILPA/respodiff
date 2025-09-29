# RespoDiff

This repository contains the **official code** for our NeurIPS 2025 paper:  
**RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation**

📄 [Paper Link](https://www.arxiv.org/abs/2509.15257) | 🌐 Project Website (coming soon)


## Overview

RespoDiff introduces a **dual-module bottleneck transformation** for diffusion-based text-to-image (T2I) generation, balancing fairness and safety with faithfulness.  
The code is adapted from [InterpretDiffusion](https://github.com/hangligit/InterpretDiffusion).


## Installation

To create the environment from the provided YAML file:

```bash
conda env create -f respodiff.yml
conda activate respodiff
```

## Training

We provide training scripts under the [`scripts/`](scripts/) directory.  
Each script corresponds to a different training configuration:

```bash
# Train with gender-related constraints
bash scripts/gender.sh
# Train with race-related constraints
bash scripts/race.sh
# Train with safety-related constraints
bash scripts/safe.sh
```

## Evaluation

We provide evaluation scripts under the [`scripts/`](scripts/) directory. The evaluation pipeline is adapted from  [Erasing Concepts in Diffusion Models](https://github.com/rohitgandikota/erasing).  
Each script corresponds to evaluating a specific aspect:

```bash
# Evaluate gender-related results
bash scripts/gender_eval.sh
# Evaluate race-related results
bash scripts/race_eval.sh
# Evaluate safety-related results
bash scripts/safe_eval.sh
```
Pretrained models for evaluation are available here: [RespoDiff pretrained models](https://drive.google.com/drive/folders/1nE7bz3t78jUyekoXQOEOGAXr04i-6jB8?usp=sharing)


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{sreelatha2025respodiffdualmodulebottlenecktransformation,
      title={RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation}, 
      author={Silpa Vadakkeeveetil Sreelatha and Sauradip Nag and Muhammad Awais and Serge Belongie and Anjan Dutta},
      year={2025},
      eprint={2509.15257},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.15257}, 
}