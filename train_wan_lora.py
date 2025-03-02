import os
import gc
import random
import numpy as np
import argparse
import json
import datetime
from tqdm import tqdm
from time import perf_counter
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file, save_file
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb

# from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
# from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
# from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from utils.temp_rng import temp_rng
from utils.dataset import CombinedDataset


def download_model(args):
    from huggingface_hub import snapshot_download
    
    # get text encoder, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "models_t5_umt5-xxl-enc-bf16.pth",
        max_workers = 1,
    )
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "google/*",
    )
    
    # get vae, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/vae",
        allow_patterns = "Wan2.1_VAE.pth",
        max_workers = 1,
    )
    
    # get clip vision if it's i2v
    if "-I2V-" in args.pretrained_model_name_or_path:
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            max_workers = 1,
        )
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "xlm-roberta-large/*",
        )
    
    # get the correct diffusion model for HF identifier
    snapshot_download(
        repo_type = "model",
        repo_id = args.pretrained_model_name_or_path,
        local_dir = "./models/" + args.pretrained_model_name_or_path,
        allow_patterns = ["config.json", "diffusion_pytorch_model*"],
        max_workers = 1,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download all necessary models to ./models if missing",
    )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
    )
    parser.add_argument(
        "--init_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to load instead of random init, must be the same rank and target layers",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = None,
        help = "Path to dataset directory with train and val subdirectories",
    )
    parser.add_argument(
        "--val_samples",
        type = int,
        default = 4,
        help = "Maximum number of videos to use for validation loss"
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training"
    )
    parser.add_argument(
        "--resolution",
        type = int,
        default = 624,
        choices=[624, 960],
        help = "Base resolution bucket, resized to equal area based on aspect ratio"
    )
    parser.add_argument(
        "--token_limit",
        type = int,
        default = 15_000,
        help = "Combined resolution/frame limit based on transformer patch sequence length: (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)"
    )
    parser.add_argument(
        "--max_frame_stride",
        type = int,
        default = 2,
        help = "1: use native framerate only. Higher values allow randomly choosing lower framerates (skipping frames to speed up the video)"
    )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 2e-4,
        help = "Base learning rate",
    )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 16,
        help = "The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
    )
    parser.add_argument(
        "--val_steps",
        type = int,
        default = 100,
        help = "Validate after every n steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 100,
        help = "Save a checkpoint of the training state every X steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 1000,
        help = "Total number of training steps",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model(args)
        exit()
    
    if args.dataset is not None:
        if args.cache_embeddings:
            cache_embeddings(args)
            exit()
        
        main(args)
    else:
        print("--dataset is required but not provided")