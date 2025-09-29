from cleanfid import fid
import argparse
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from utils_model import save_model, load_model
from datasets import load_dataset
from model import model_types
from PIL import Image
import clip
import torch
import pandas as pd
import os
import re
import json
import copy
import random
from utils_data import parse_concept
from tqdm.auto import tqdm
from torch.multiprocessing import Pool, set_start_method
from utils_data import int_to_onehot
from diffusers.utils import logging
logging.disable_progress_bar() 

# taken from https://github.com/rohitgandikota/erasing/tree/main/eval-scripts

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def initialize_model(args):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    
    scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False,
            num_train_timesteps=1000,
            steps_offset=1,
        )
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    mlp=model_types[args.target_model_type](resolution=args.resolution//64)
    unet.set_controlnet(mlp)
    mlp=model_types[args.residual_model_type](resolution=args.resolution//64)
    unet.set_controlnet_residual(mlp)
    unet.requires_grad_(False)
    
    if args.concept:
        unet_concept = copy.deepcopy(unet)
        concept_dict = json.load(open('concept_dict.json','r'))
        for each_concept in args.concept:
            concept_idx = concept_dict[each_concept]
            load_model(unet_concept, os.path.join(args.checkpoint, each_concept, str(args.seed), 'unet.pth'))
            unet.controlnet.fc1.weight[:, concept_idx] = unet_concept.controlnet.fc1.weight[:, concept_idx]
            unet.controlnet_residual.fc2.weight[:, concept_idx] = unet_concept.controlnet_residual.fc2.weight[:, concept_idx]
        del unet_concept
        
    if args.task == 'safe':
        sexual_concept_idx = concept_dict['sexual']
        violence_concept_idx = concept_dict['violence']
        unet.controlnet.fc1.weight[:, sexual_concept_idx] =  unet.controlnet.fc1.weight[:, sexual_concept_idx] + unet.controlnet.fc1.weight[:, violence_concept_idx]
        unet.controlnet_residual.fc2.weight[:, sexual_concept_idx] =  unet.controlnet_residual.fc2.weight[:, sexual_concept_idx] + unet.controlnet_residual.fc2.weight[:, violence_concept_idx]
        args.concept='sexual'
        
    
    model=StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    if args.fp16:
        # print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

    return model


def generate_i2p(batch_args):
    global_start_idx, batch_prompts, concept_labels, gpu_id, args = batch_args
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    model = initialize_model(args).to(device)
    model.set_progress_bar_config(disable=True)
    
    concept_dict = json.load(open('concept_dict.json', 'r'))
    concept = int_to_onehot(concept_dict[args.concept], 10).to(device).unsqueeze(0)
    
    num_samples = args.num_samples_per_prompt
    
    expanded_prompts = [prompt for prompt in batch_prompts for _ in range(num_samples)]
    expanded_concept_labels = [batch_labels for batch_labels in concept_labels for _ in range(num_samples)]
    batch_generators = [torch.Generator("cuda").manual_seed(global_start_idx + i) for i in range(len(expanded_prompts))]  # Generate one seed per image
    
    images = predict_cond(model=model, prompt=expanded_prompts, generator=batch_generators, condition=concept)
    
    saved_labels = []
    saved_prompts = []
    for i, (image, label, prompt) in enumerate(zip(images, expanded_concept_labels, expanded_prompts)):
        global_image_idx = global_start_idx + i
        image.save(os.path.join(args.im_dir, f"{global_image_idx}.jpg"))
        saved_labels.append(label) 
        saved_prompts.append(prompt)
        
    torch.cuda.empty_cache()
    return len(images), saved_labels, saved_prompts



def predict_cond(model, 
                 prompt, 
                 generator,
                 condition, 
                 num_inference_steps=50,
                 interpolator=None
                 ):

    output = model(prompt=prompt,
                   generator=generator,
                   controlnet_cond=condition,
                   num_inference_steps=num_inference_steps,
                   controlnet_interpolator=interpolator,
                   )
    image = output[0]
    return image

def create_images(args, prompts, concept_labels):
    args.num_samples_per_prompt = 5  
    total_samples = len(prompts) * args.num_samples_per_prompt
    
    batch_args = []
    batch_size = args.batch_size  
    global_start_idx = 0 
    gpu_id = 0
    
    for idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[idx: idx + batch_size]
        batch_concept_labels = concept_labels[idx: idx + batch_size]
        batch_args.append((global_start_idx, batch_prompts, batch_concept_labels, gpu_id, args))
        global_start_idx += len(batch_prompts) * args.num_samples_per_prompt
        gpu_id = (gpu_id + 1) % args.num_gpus
 
 
    set_start_method('spawn', force=True)

    saved_labels = []
    saved_prompts = []

    with Pool(processes=args.num_gpus) as pool:
        print(f"Generating {total_samples} images across {args.num_gpus} GPUs...")
        results = list(
            tqdm(pool.imap_unordered(generate_i2p, batch_args), total=len(batch_args), desc="Generating I2P images")
        )
        
        for _, batch_labels, batch_prompts in results:
            saved_labels.extend(batch_labels)
            saved_prompts.extend(batch_prompts)
           
            

    print(f"Image generation complete. {len(saved_labels)} images saved in {args.im_dir}")
    return saved_labels, saved_prompts


def evaluate_q16_nudity(args, prompts, labels):
    from metrics.nudenet_classify import detect_nude_and_q16
    predictions=detect_nude_and_q16(folder=args.im_dir)
    logging=pd.DataFrame({'prompt':prompts, 'label':labels, 'prediction':predictions})
    logging = logging.explode('label').reset_index(drop=True)
    logging.to_csv(args.root_dir + '/i2p.csv', index=False)
    i2p_mean = pd.read_csv(args.root_dir + '/i2p.csv').prediction.mean()
    metrics = {'i2p_mean' : i2p_mean}
    with open(args.root_dir + '/i2p_mean_stats.json', 'w') as fp:
            json.dump(metrics, fp)
    stats=pd.read_csv(args.root_dir + '/i2p.csv').groupby("label").prediction.mean()
    stats.to_csv(args.root_dir + '/i2p_individual_category_stats.csv', index=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Q16/Nudity Classification')
    parser.add_argument('--root_dir', type=str, required=True) 
    parser.add_argument('--task', type=str, required=True, default='safe') 
    parser.add_argument('--fp16', action='store_true', help="use float16 precision")
    parser.add_argument('--concept', nargs='+') 
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples_per_prompt', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--revision",type=str,default=None,required=False,)
    parser.add_argument("--model_type",type=str,default="MLP")
    parser.add_argument('--checkpoint', help='model to analyse', type=str, required=True)
    parser.add_argument("--target_model_type",type=str,default="MLP_target")
    parser.add_argument("--residual_model_type",type=str,default="MLP_residual")
    parser.add_argument("--resolution",type=int,default=512)

    
    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, 'q16_nudity', str(args.task))
    args.im_dir= os.path.join(args.root_dir, 'images')
    os.makedirs(args.im_dir, exist_ok=True)
    prompt_dataframe = load_dataset("AIML-TUDA/i2p")['train'].to_pandas()
    prompts= prompt_dataframe['prompt'].tolist()
    concept_labels = [label.replace(" ","").split(',') for label in list(prompt_dataframe.categories)]
    labels, prompts = create_images(args, prompts, concept_labels)
    evaluate_q16_nudity(args, prompts, labels)
    
