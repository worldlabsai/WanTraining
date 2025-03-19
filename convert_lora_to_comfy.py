import os
import argparse
import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_lora",
        type = str,
        required = True,
        help = "Path to LoRA .safetensors",
    )
    parser.add_argument(
        "--alpha",
        type = float,
        default = None,
        help = "Optional alpha value, defaults to rank",
    )
    parser.add_argument(
        "--dtype",
        type = str,
        default = None,
        help = "Optional dtype (bfloat16, float16, float32), defaults to input dtype",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    original_lora_sd = load_file(args.input_lora)
    converted_lora_sd = {}
    for key in original_lora_sd.keys():
        new_key = "diffusion_model." + key
        
        if "patch_embedding.lora_A" in new_key:
            print("control lora detected")
            
            new_ch = original_lora_sd[key].shape[1] # 32
            model_dim = original_lora_sd[key.replace(".lora_A.weight", ".lora_B.weight")].shape[0]
            
            reshape_weight = torch.tensor([model_dim, new_ch, 1, 2, 2])
            reshape_key = new_key.replace(".lora_A.weight", ".reshape_weight")
            converted_lora_sd[reshape_key] = reshape_weight
            
            print(f"added reshape_weight for {reshape_key}: {reshape_weight}")
        
        converted_lora_sd[new_key] = original_lora_sd[key]
    
    if args.alpha is not None:
        for key in list(converted_lora_sd.keys()):
            if "lora_A" in key:
                alpha_name = key.replace(".lora_A.weight", ".alpha")
                converted_lora_sd[alpha_name] = torch.tensor([args.alpha], dtype=converted_lora_sd[key].dtype)

    dtype = None
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32

    if dtype is not None:
        dtype_min = torch.finfo(dtype).min
        dtype_max = torch.finfo(dtype).max
        for key in converted_lora_sd.keys():
            if converted_lora_sd[key].min() < dtype_min or converted_lora_sd[key].max() > dtype_max:
                print(f"warning: {key} has values outside of {dtype} {dtype_min} {dtype_max} range")
            converted_lora_sd[key] = converted_lora_sd[key].to(dtype)
    
    output_path = os.path.splitext(args.input_lora)[0] + "_comfy.safetensors"
    save_file(converted_lora_sd, output_path)
    print(f"saved to {output_path}")