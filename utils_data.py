import glob
import os
import json
from PIL import Image
import numpy as np
import torch
import random
import pandas as pd

def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]


def int_to_onehot(x, n, weights=None):

    if not isinstance(x, list):
        x = [x]
    assert isinstance(x[0], int)
    x = torch.tensor(x).long()
    v = torch.zeros(n)
    if weights is not None:
        if not isinstance(weights, list):
            weights = [weights] * len(x)  
        assert len(weights) == len(x)
        for i, w in zip(x, weights):
            v[i] = w
    else:
        v[x] = 1. 

    return v

# def int_to_onehot(x, n):
#     if not isinstance(x, list):
#         x = [x]
#     assert isinstance(x[0], int)
#     x = torch.tensor(x).long()
#     v = torch.zeros(n)
#     v[x] = 1.
#     return v

def parse_concept(input_concept):
    """
    parse the input concept into a list of concepts for evaluation, supported formats:
    concept: str:
        'man' -> ['man'] (generate an image of a man)
    concept: str:
        'man,young' -> ['man', 'young'] (generate an image of a young man)
    concept: list[str]:
        ['man', 'woman'] -> ['man'], ['woman'] (generate two images: a man, a woman)
    concept: list[str]:
        ['man,young', 'woman,young'] -> [['man','young'],['woman','young']] (generate two images: a young man, an old woman)

    The output of this function is directly fed to int_to_onehot and return a multi-hot vector which can be directly used by the model
    """
    def parse_concept_string(concept):
        assert isinstance(concept, str)
        concept = concept.split(',')
        concept = [x.strip() for x in concept]
        return concept
    
    if isinstance(input_concept, str):
        input_concept = parse_concept_string(input_concept)
        input_concept = [input_concept]

    elif isinstance(input_concept, list):
        input_concept = [parse_concept_string(x) for x in input_concept]

    else:
        raise ValueError(input_concept)
    
    return input_concept
    
    
def get_test_data(given_prompt=None, given_concept=None, with_baseline=True, device='cuda', max_concept_length=100):
    """
    data_dir: path to data file
    prompt: str
    concept: str or list[str]
    """
    if given_prompt:
        prompt=given_prompt
    else:
        prompt = ['temp']
    if given_concept:
        concept_dict=json.load(open('concept_dict.json','r'))
        concept = parse_concept(given_concept)
        print(f'eval with concept: {concept}')
        concept=[int_to_onehot([concept_dict[c_i] for c_i in c], max_concept_length).to(device).unsqueeze(0) for c in concept]
        if with_baseline:
            concept.insert(0, None)
        prompt = [prompt] * len(concept)
        return prompt, concept
    return None


def get_i2p_data(given_prompt=None, given_concept=None, with_baseline=False, device='cuda', max_concept_length=100):
    import pandas as pd
    i2p = pd.read_csv("hf://datasets/AIML-TUDA/i2p/i2p_benchmark.csv")
    if given_prompt:
        prompts=i2p[i2p.categories.apply(lambda x: given_prompt in x)].prompt.values.tolist()
    else:
        prompts = i2p.prompt.values.tolist()

    concept_label = [label.replace(" ","").split(',') for label in list(i2p.categories)]
    # concept_label=[given_concept] if isinstance(given_concept, str) else given_concept
    concept_dict=json.load(open('concept_dict.json','r'))
    concept=int_to_onehot(concept_dict[given_concept], max_concept_length).to(device).unsqueeze(0)
    if with_baseline:
        concept.insert(0, None)
        concept_label.insert(0, 'none')

    inputs = []
    for prompt, label in zip(prompts, concept_label):
            inputs.append([prompt, concept, label])
    return inputs

def get_coco30k_data(real_img_dir, sample=False):
    
    os.makedirs(real_img_dir, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("sayakpaul/coco-30-val-2014")
    if sample:
        sample_5k = random.sample(range(0, 30000), 5000)
        ds = ds['train'][sample_5k]
    else:
        ds = ds['train']
    for idx, img in enumerate(ds['image']):
        img.save(f"{real_img_dir}/{idx}.jpg")
    dict = {'captions' : ds['caption']}
    df = pd.DataFrame(dict) 
    df.to_csv(f'{real_img_dir}/prompts.csv', index=False)
    