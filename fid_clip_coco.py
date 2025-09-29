
from cleanfid import fid
import argparse
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from utils_model import save_model, load_model
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
from itertools import product
from utils_data import parse_concept
from tqdm.auto import tqdm
from torch.multiprocessing import Pool, set_start_method
from utils_data import int_to_onehot
from diffusers.utils import logging
logging.disable_progress_bar() 

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

def generate_coco_safe(batch_args):
    start_idx, batch_prompts, gpu_id, args = batch_args
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    model = initialize_model(args).to(device)
    model.set_progress_bar_config(disable=True)
    
    concept_dict = json.load(open('concept_dict.json','r'))
    concept = int_to_onehot(concept_dict[args.concept], 10).to(device).unsqueeze(0)
    generators = [torch.Generator("cuda").manual_seed(seed) for seed in range(start_idx, start_idx + len(batch_prompts))]
    images=predict_cond(model=model, prompt=batch_prompts, generator=generators, condition=concept)

    for i, image in enumerate(images):
            image_idx = start_idx + i
            image.save(os.path.join(args.im_dir, f"{image_idx}.jpg"))
    torch.cuda.empty_cache()
    return len(images)

            
def generate_coco_fair(batch_args):
    start_idx, batch_prompts, gpu_id, args = batch_args
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    model = initialize_model(args).to(device)
    
    concept_dict=json.load(open('concept_dict.json','r'))
    concept = parse_concept(args.concept)
    model.set_progress_bar_config(disable=True)
    select_concept = [random.choice(concept)]
    if args.compose :
        compose_concepts = [['a man', 'a woman'], ['a black-race person', 'a white-race person', 'a Asian-race person']]
        select_concept = [random.choice(list(product(*compose_concepts)))]
    select_concept=[int_to_onehot([concept_dict[c_i] for c_i in c], 10).to(device).unsqueeze(0) for c in select_concept][0]
    
    args.num_test_samples = 5
    generators = [torch.Generator("cuda").manual_seed(seed) for seed in range(start_idx, start_idx + len(batch_prompts))]
    images=predict_cond(model=model, prompt=batch_prompts, generator=generators, condition=select_concept)
    
    for i, image in enumerate(images):
            image_idx = start_idx + i
            image.save(os.path.join(args.im_dir, f"{image_idx}.jpg"))
    torch.cuda.empty_cache()
    return len(images)


def create_images(args, orig_prompts):
   
        if os.path.exists(args.im_dir):
            existing_files = [f for f in os.listdir(args.im_dir) if f.endswith('.jpg')]
            existing_indices = [
                int(re.match(r"(\d+)\.jpg", f).group(1)) for f in existing_files if re.match(r"(\d+)\.jpg", f)
            ]
            start_index = max(existing_indices) + 1 if existing_indices else 0
        else:
            start_index = 0
            
        batch_args = []
        batch_size=args.batch_size
        
        gpu_id=0
        for idx in range(start_index, len(orig_prompts), batch_size):
            batch_prompts = prompts[idx: idx + batch_size]
            batch_args.append((idx, batch_prompts, gpu_id, args))
            gpu_id = (gpu_id + 1) % args.num_gpus  
        
        if args.task == 'gender' or args.task == 'race':
            # Set start method for multiprocessing
            set_start_method('spawn', force=True)

            # Use multiprocessing to generate images in parallel
            with Pool(processes=args.num_gpus) as pool:
                print(args.num_gpus)
                results = list(
                    tqdm(pool.imap_unordered(generate_coco_fair, batch_args), total=len(batch_args), desc="Generating COCO30K images")
                )
        elif args.task == 'safe':
            # Set start method for multiprocessing
            set_start_method('spawn', force=True)

            # Use multiprocessing to generate images in parallel
            with Pool(processes=args.num_gpus) as pool:
                print(args.num_gpus)
                results = list(
                    tqdm(pool.imap_unordered(generate_coco_safe, batch_args), total=len(batch_args), desc="Generating COCO30K images")
                )

        print(f"Image generation complete. {sum(results)} images saved in {args.im_dir}")
        

        
def evaluate_fid(args):
    metrics ={}
    save_path = os.path.join(args.root_dir, 'fid_coco30k.csv')
    print(args.im_dir)
    fid_score = fid.compute_fid(args.coco_src_img_dir, args.im_dir, num_workers=1)
    print('FID : ',fid_score)
    metrics.update({'fid-score' : fid_score})
    with open(save_path, 'w') as fp:
        json.dump(metrics, fp)
        
def evaluate_clip(args, prompts):
    device=torch.device('cuda')
    metrics ={}
    save_path = os.path.join(args.root_dir, 'clip_coco30k.csv')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    im_folder = args.im_dir
    ratios = {}
    
    images = os.listdir(im_folder)
    images = [im for im in images if '.png' in im or '.jpg' in im]
    images = sorted_nicely(images)
    for image in tqdm(images, total=len(images)):
        case_number = int(image.split('_')[0].replace('.jpg',''))
        caption = prompts.iloc[case_number].iloc[0]
        im = Image.open(os.path.join(im_folder, image))
        inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        clip_score = outputs.logits_per_image[0][0].detach().cpu()
        ratios[case_number] = ratios.get(case_number, []) + [clip_score]
        
    clip_df = pd.DataFrame.from_dict(ratios, orient='index',columns=['clip'])
    clip_df.to_csv(f'{args.root_dir}/individual_clip_coco30k.csv')
    mean_clip = clip_df['clip'].mean()
    
    print('CLIP : ', mean_clip)
    
    metrics.update({'mean-clip-score' : mean_clip})
    with open(save_path, 'w') as fp:
        json.dump(metrics, fp)
    
   
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'FID CLIP evaluation')
    parser.add_argument('--coco_src_img_dir', type=str, default='data/coco30k') 
    parser.add_argument('--root_dir', type=str, required=True) 
    parser.add_argument('--task', type=str, required=True, default='gender') 
    parser.add_argument('--fp16', action='store_true', help="use float16 precision")
    parser.add_argument('--compose', action='store_true', help="use for composition")
    parser.add_argument('--concept', nargs='+') 
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--revision",type=str,default=None,required=False,)
    parser.add_argument("--model_type",type=str,default="MLP")
    parser.add_argument('--checkpoint', help='model to analyse', type=str, required=True)
    parser.add_argument("--target_model_type",type=str,default="MLP_target")
    parser.add_argument("--residual_model_type",type=str,default="MLP_residual")
    parser.add_argument("--resolution",type=int,default=512)

    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, 'fid-clip-coco', str(args.task))
    args.im_dir= os.path.join(args.root_dir, 'images')
    os.makedirs(args.im_dir, exist_ok=True)
    prompt_dataframe = pd.read_csv(os.path.join(args.coco_src_img_dir, 'prompts.csv'))
    prompts= prompt_dataframe['captions'].tolist()
    create_images(args, prompts)
    evaluate_fid(args)
    evaluate_clip(args, prompt_dataframe)