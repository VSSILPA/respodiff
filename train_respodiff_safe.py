import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from utils_data import int_to_onehot
from torchvision import transforms
import torch.nn.functional as F 

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from model import model_types
from config import parse_args
from utils_model import save_model, load_model
from utils_data import  get_test_data

import wandb 
from PIL import Image

logger = get_logger(__name__)


def unfreeze_layers_unet(unet):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)


def get_denoised_images(model, latents, prompt, sample_till_t, condition=None, seed=None):
    
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    prompt = [prompt]    
    latent = model(prompt=prompt, generator=generator, latents=latents, guidance_scale=3, num_inference_steps=50, output_type='latent', controlnet_cond=condition, total_timesteps=sample_till_t)
    return latent[0]

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = 256, 256
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        img = img.resize((w, h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def predict_cond(model, prompt, generator,condition,num_inference_steps=50, projector=None):

        output = model(prompt=prompt,
                        generator=generator,
                        controlnet_cond=condition,
                        projector=projector,
                        num_inference_steps=num_inference_steps)
        image = output[0]
        return image
    
def log_validation(model, target_onehot, args, accelerator):
    logger.info("Running validation... ")

    model.set_progress_bar_config(disable=True)
    
    grid_images = []
    batch_prompts= [args.valid_prompt] * args.num_valid_data
    
    generators = [torch.Generator("cuda").manual_seed(seed+13678) for seed in range(0, args.num_valid_data)]
    grid_images.extend(predict_cond(model=model, prompt=batch_prompts, generator=generators, condition=None))
    generators = [torch.Generator("cuda").manual_seed(seed+13678) for seed in range(0, args.num_valid_data)]
    grid_images.extend(predict_cond(model=model, prompt=batch_prompts, generator=generators, condition=target_onehot))
    generators = [torch.Generator("cuda").manual_seed(seed+13678) for seed in range(0, args.num_valid_data)]
    grid_images.extend(predict_cond(model=model, prompt=batch_prompts, generator=generators, condition=target_onehot, projector=lambda x: 0*x))
    
    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            plt = image_grid(grid_images, rows=3, cols=args.num_valid_data)
            tracker.log({"validation_images for prompt " + args.valid_prompt: wandb.Image(plt)})
        
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")
            
    del model
    torch.cuda.empty_cache()


def main():
    
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    device=accelerator.device
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

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
    ).to(device)
    
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    model=StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)

    if args.use_esd:
        load_model(unet, 'baselines/diffusers-nudity-ESDu1-UNET.pt')

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    mlp=model_types[args.target_model_type](resolution=args.resolution//64)
    unet.set_controlnet(mlp)
    mlp=model_types[args.residual_model_type](resolution=args.resolution//64)
    unet.set_controlnet_residual(mlp)
    unet = unfreeze_layers_unet(unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_c = torch.optim.Adam(
        unet.controlnet.fc1.parameters(),
        lr=args.c_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    optimizer_s = torch.optim.Adam(
        unet.controlnet_residual.fc2.parameters(),
        lr=args.s_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    if args.fp16:
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()
        unet=unet.half()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
   
    print('weight_dtype',weight_dtype)

    model.to(device)
    unet.to(device)

    if accelerator.is_main_process:
        exp_name = args.output_dir
        accelerator.init_trackers(
            project_name="respodiff", 
            config={k:v for k,v in vars(args).items() if k!='config'},
            init_kwargs={"wandb": {"name": exp_name}}
            )

    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            tracker.run.log_code(".")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}") 
            

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    
    pbar = tqdm(range(args.max_train_steps))
    
    target_prompt = args.concept
    neutral_prompt = args.neutral_concept

    concept_dict=json.load(open('concept_dict.json','r'))
    if len(target_prompt) == 2:
        target_onehot = int_to_onehot(concept_dict[target_prompt[1]], 10).unsqueeze(0).to(device)
    else:
        target_onehot = int_to_onehot(concept_dict[target_prompt], 10).unsqueeze(0).to(device)
    
    model, unet, optimizer_c, optimizer_s = accelerator.prepare(model, unet, optimizer_c, optimizer_s) 

    print("Start training")
    loss_history=[]

    for step in pbar:
        unet.train()
        
        optimizer_c.zero_grad()
        optimizer_s.zero_grad()
        noise_scheduler.set_timesteps(args.ddim_steps, device=device)
        t_enc = torch.randint(args.start_ddim_steps, args.end_ddim_steps, (args.train_batch_size, )).to(device)
        noise_scheduler.set_timesteps(1000)
        t_enc_ddpm = torch.LongTensor([noise_scheduler.timesteps[int(t_enc[i] * args.ddpm_steps / args.ddim_steps)] for i in range(len(t_enc))]).to(device)
        t_enc_ddpm = t_enc_ddpm.long()
        latent = []
        latent_neutral = []
        
        for timestep in t_enc:
            latent = torch.randn((1, 4, 64, 64)).to(weight_dtype).to(device)
            latent_neutral.append(get_denoised_images(model, latent, neutral_prompt, int(timestep), condition=target_onehot))
        latent_neutral = torch.stack(latent_neutral).view(-1, 4, 64, 64)
        
        with torch.no_grad(): 
            t_enc_ddpm = torch.cat([t_enc_ddpm] * 2)
            latent_model_input = torch.cat([latent_neutral] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t_enc_ddpm)
            
            encoder_hidden_states = model._encode_prompt(target_prompt[0], device, negative_prompt = target_prompt[1], num_images_per_prompt = args.train_batch_size, do_classifier_free_guidance=True)
            noise_pred = unet(latent_model_input, t_enc_ddpm, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            eps_target = noise_pred_uncond + args.target_guidance_scale * (noise_pred_text - noise_pred_uncond)
            

        latent_model_input = torch.cat([latent_neutral] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t_enc_ddpm)
        
        encoder_hidden_states = model._encode_prompt(neutral_prompt, device, num_images_per_prompt = args.train_batch_size, do_classifier_free_guidance=True)
        output = unet(latent_model_input, t_enc_ddpm, encoder_hidden_states=encoder_hidden_states, controlnet_cond=target_onehot)
        noise_pred = output.sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        eps_neutral_with_target = noise_pred_uncond + args.neutral_guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        eps_target.requires_grad = False
        concept_loss = F.mse_loss(eps_neutral_with_target, eps_target, reduction="mean")
        
        accelerator.backward(concept_loss, retain_graph=True)
        optimizer_c.step()
        
        output = unet(latent_model_input, t_enc_ddpm, encoder_hidden_states=encoder_hidden_states, controlnet_cond=target_onehot, controlnet_residual=target_onehot)
        noise_pred = output.sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        eps_neutral_with_target_residual = noise_pred_uncond + args.neutral_guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        output = unet(latent_model_input, t_enc_ddpm, encoder_hidden_states=encoder_hidden_states)
        noise_pred = output.sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        eps_neutral_unmodified = noise_pred_uncond + args.neutral_guidance_scale * (noise_pred_text - noise_pred_uncond)

        alignment_loss = args.lambda_align * F.mse_loss(eps_neutral_with_target_residual, eps_neutral_unmodified, reduction="mean")
        
        accelerator.backward(alignment_loss)
        optimizer_s.step()

        loss = concept_loss + alignment_loss
        train_loss = loss.item()
        
        pbar.set_postfix({"loss": loss.item()})
        
        accelerator.log({"concept_loss": concept_loss.item()}, step=step)
        accelerator.log({"alignment_loss": alignment_loss.item()}, step=step) 

        accelerator.log({"train_loss": train_loss}, step=step)
        loss_history.append(train_loss)

        if not args.skip_evaluation and (step)%args.log_every_steps==0:
                save_model(unet, args.output_dir+'/unet_' + str(step) + '.pth')
                plt.figure()
                plt.plot(loss_history)
                plt.savefig(args.output_dir+'/loss_history.png')
                plt.close()
                log_validation(model, target_onehot, args, accelerator)

    save_model(unet, args.output_dir+'/unet.pth')
    load_model(unet, args.output_dir + '/unet.pth') 
    log_validation(model, target_onehot, args, accelerator) 
    plt.figure()
    plt.plot(loss_history)
    plt.savefig(args.output_dir+'/loss_history.png')
    plt.close()
 

if __name__ == "__main__":
    main()