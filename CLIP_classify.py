from PIL import Image

import os
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import torch
import json
from utils_model import save_model, load_model
from model import model_types
from itertools import product
import copy
from tqdm import tqdm
from winobias_cfg import professions, templates
from utils_data import int_to_onehot
from torch.multiprocessing import Pool, set_start_method
from diffusers.utils import logging
logging.disable_progress_bar() 

# taken from https://github.com/rohitgandikota/erasing/tree/main/eval-scripts


def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt

def add_winobias_metrics(df):
    def deviation_ratio(list_of_counts):
        r = np.array(list_of_counts)
        r=r/np.sum(r)
        ref=np.ones((len(r)))/len(r)
        return np.abs(r-ref).max()/(1-1/len(r))
    columns = df.columns
    df['deviation_ratio'] = df.apply(lambda row: deviation_ratio(row[columns]), axis=1)
    return df

def predict_cond(model, 
                 prompt, 
                 condition, 
                 num_test_samples,
                 base_seed,
                 num_inference_steps=50,
                 interpolator=None
                 ):
    
    def get_inputs(prompt=prompt, batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(base_seed + i) for i in range(batch_size)]
        prompts = batch_size * [prompt]
        return {"prompt": prompts, "generator":generator}

    output = model(**get_inputs(prompt=prompt, batch_size=num_test_samples),
                   controlnet_cond=condition,
                   num_inference_steps=num_inference_steps,
                   controlnet_interpolator=interpolator,
                   )
    image = output[0]
    return image


def process_profession(profession, args, templates, model_name, gpu_id):
    """Generate and save images for a single profession on a specific GPU."""
    # Set the specific GPU for this process
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Processing profession: {profession} on {device}")

    # Load the model onto the assigned GPU
    model = model_name.to(device)

    save_image_dir = os.path.join(args.im_dir, profession)
    os.makedirs(save_image_dir, exist_ok=True)

    template_lst = templates[args.template_key]
    prompts = [prompt_with_template(profession, temp) for temp in template_lst]

    
    concept_dict = json.load(open('concept_dict.json','r'))
    conditions =[int_to_onehot(concept_dict[c], 10).to(args.device).unsqueeze(0) for c in args.concept]
    if args.compose :
        compose_concepts = [[ 'a woman', 'a man'], ['a black-race person', 'a white-race person', 'a Asian-race person']]
        gender_concepts = ['a woman', 'a man']
        race_concepts = ['a black-race person', 'a white-race person', 'a Asian-race person']
        gender_scale = args.gender_scale  # e.g., 1.0
        race_scale = args.race_scale      # e.g., 0.5
        conditions = []
        for c in product(*compose_concepts):
            indices = [concept_dict[c_i] for c_i in c]
            weights = [gender_scale if c_i in gender_concepts else race_scale for c_i in c]
            vec = int_to_onehot(indices, 10, weights=weights).to(args.device).unsqueeze(0)
            conditions.append(vec)
        args.num_test_samples = 5

    global_idx = 0
    for prompt in prompts:
        print(f'Creating images with prompt: {prompt}')
        for concept in conditions:
                images = predict_cond(
                    model=model,
                    prompt=prompt,
                    condition=concept,
                    num_test_samples=args.num_test_samples,
                    base_seed=global_idx
                )
                for image in images:
                    image.save(os.path.join(save_image_dir, f"{global_idx}.jpg"))
                    global_idx = global_idx + 1
             

def generate_images(args):
 
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
            # load_model(unet_concept, os.path.join(args.checkpoint, each_concept, str(args.t_dir), 'unet.pth'))
            unet.controlnet.fc1.weight[:, concept_idx] = unet_concept.controlnet.fc1.weight[:, concept_idx]
            unet.controlnet_residual.fc2.weight[:, concept_idx] = unet_concept.controlnet_residual.fc2.weight[:, concept_idx]
        del unet_concept
    


      
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
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

        
    print(f'images saved to: {args.im_dir}')  
    
    available_gpus = args.num_gpus
    num_workers = min(len(professions), available_gpus)  
    gpu_assignments = list(range(num_workers))

    
    set_start_method('spawn', force=True)
    
    tasks = [
            (profession, args, templates, model, gpu_assignments[i % available_gpus])
            for i, profession in enumerate(professions)]
    
    for i in tqdm(range(0, len(tasks), available_gpus),  desc="Processing professions"):
        current_batch = tasks[i:i + available_gpus]
    
        with Pool(processes=min(len(current_batch), available_gpus)) as pool:
            
            results = list(
                    tqdm(pool.starmap(process_profession,  current_batch), total=len(current_batch), desc="Processing professions")
                )
            print(f"Batch {i // available_gpus + 1} done:")
            
        print("All professions processed.")
    
def CLIP_classification_function(im_dir, imgs, attributes, model, processor, from_case=0, till_case=1000000, return_df=False, args=None):
    
    ratios = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logits_per_image = []
    for image in imgs:
        try:
            case_number = int(image.split('_')[0].replace('.png','').replace('.jpg',''))
            if case_number < from_case or case_number > till_case:
                continue

            im = Image.open(os.path.join(im_dir, image))
            
            inputs = processor(text=attributes, images=im, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            if args.eval_type=='winobias-alignment':
                logits_per_image.append(outputs.logits_per_image.item()) 
            else:
                logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu()  
            tmax = probs.max(1, keepdim=True)[0]
            mask = list(probs.ge(tmax)[0].float().numpy())
            ratios[case_number] = ratios.get(case_number, []) + [mask]
        except Exception:
            ratios[case_number] = ratios.get(case_number, []) + [[0]*len(attributes)]
            
    if args.eval_type=='winobias-alignment':
        return logits_per_image
    else:
        columns = [f"{att.replace(' ','_')}_bias_cnt" for att in attributes]
        df=pd.DataFrame()
        for col in columns:
            df[col] = np.nan
        for key in ratios.keys():
            for idx, col in enumerate(columns):
                df.loc[key,col] = np.mean(np.array(ratios[key])[:,idx])
        if return_df:
            return df
        if not len(columns)==2: print("WARNING - CLIP_classification_function: #attributes is not 2")
        return df.mean().tolist()[0]
        


def evaluate_winobias(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_attributes = args.clip_attributes
    logging = []
    
    im_dir = args.im_dir
    professions = os.listdir(im_dir)
    professions = sorted([prof for prof in professions if not '.csv' in prof])
 
    
    for profession in professions:  
        im_dir = os.path.join(args.im_dir, profession)
        images = os.listdir(im_dir)
        images = [im for im in images if '.png' in im or '.jpg' in im]
        images = sorted_nicely(images)
        df = CLIP_classification_function(im_dir = im_dir, imgs=images, attributes=clip_attributes, model=clip_model, processor=processor, return_df=True, args=args)
        result = {'profession': profession}
        sums = df.sum().to_dict()
        result.update(sums)
        logging.append(result)
        print(result)
        
    logging = pd.DataFrame(logging)
    logging = add_winobias_metrics(logging.set_index('profession'))
    logging.loc['mean'] = logging.mean()
    
    save_name = 'winobias_dev_ratio_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(args.im_dir, save_name)
    logging.to_csv(save_path, index=True)
    print(f'CLIP classification results saved to {save_path}')
    print(f'Mean CLIP classification results: {logging.loc["mean"].to_dict()}')

def evaluate_alignment(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_scores = []
    
   
    for profession in professions:
        template_lst = templates[args.template_key]
        im_dir = os.path.join(args.im_dir, profession)
        prompts = [prompt_with_template(profession, temp) for temp in template_lst]
        for i, prompt in enumerate(prompts):
            total_samples_per_prompt = args.num_test_samples*len(args.concept)
            print(f'creating images with prompt: {prompt}')
            imgs = sorted_nicely(os.listdir(im_dir))[i*total_samples_per_prompt:(i+1)*total_samples_per_prompt]
            scores = CLIP_classification_function(im_dir=im_dir, imgs=imgs, attributes=prompt, model=clip_model, processor=processor, return_df=True, args=args)
            clip_scores.extend(scores)
            print('done')
    avg_clip_score = sum(clip_scores)/len(clip_scores)
    alignment_dict = {'winobias_clip_alignmnet': avg_clip_score}
    df = pd.DataFrame(alignment_dict, index=[0])
    save_name = 'winobias_alignment_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_path = os.path.join(args.im_dir, save_name + '_result.csv')
    df.to_csv(save_path, index=False)
    print(f'Winobias clip alignment results saved to {save_path}')
    print(avg_clip_score)
    # wandb.log(alignment_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CLIP classification',
                    description = 'Takes the path to images and gives CLIP classification scores')
    parser.add_argument('--root_dir', help='dir to folders of winobias images', type=str, required=True)
    parser.add_argument('--checkpoint', help='model to analyse', type=str, required=True)
    parser.add_argument('--clip_attributes', type=str, required=True, nargs='+')
    parser.add_argument('--eval_type', type=str, default=None, choices=['both', 'winobias', 'winobias-alignment'])
    parser.add_argument('--template_key', type=str, default="0")
    parser.add_argument('--num_test_samples', type=int, default=15, help="15 for gender, 10 for race")
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--gender_scale', type=float, default=0.6)
    parser.add_argument('--race_scale', type=float, default=0.4)
    parser.add_argument('--concept', nargs='+') 
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--revision",type=str,default=None,required=False,)
    parser.add_argument("--model_type",type=str,default="MLP")
    parser.add_argument('--compose', action='store_true', help="use for composition")
    parser.add_argument('--fp16', action='store_true', help="use float16 precision")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--target_model_type",type=str,default="MLP_target")
    parser.add_argument("--residual_model_type",type=str,default="MLP_residual")
    parser.add_argument("--resolution",type=int,default=512)
    
    args = parser.parse_args()
    
    args.im_dir= os.path.join(args.root_dir + 'winobias', str(args.concept))
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.im_dir = os.path.join(args.im_dir, 'images',  f'template{str(args.template_key)}')
    os.makedirs(args.im_dir, exist_ok=True)
    generate_images(args)
    args.eval_type='winobias'
    evaluate_winobias(args)
    args.eval_type='winobias-alignment'
    evaluate_alignment(args)
    
 