import os
import argparse
import torch
from glob import glob
from tqdm.auto import tqdm
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        required = True,
        help = "Folder with multiple safetensor checkpoints",
    )
    parser.add_argument(
        "--last_n",
        type = int,
        default = 2,
        help = "How many checkpoints to merge",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    lora_checkpoints_list = glob(os.path.join(args.checkpoint_dir, "*.safetensors"))
    
    progress_bar = tqdm(range(0, args.last_n), leave=True)
    
    sd = load_file(lora_checkpoints_list[-1])
    
    progress_bar.update(1)
    
    for i in range(args.last_n - 1):
        new_sd = load_file(lora_checkpoints_list[-2 - i])
        factor = 1 / (i + 2)
        
        diff = 0
        for key in sd.keys():
            diff += torch.nn.functional.mse_loss(sd[key], new_sd[key]).item()
            sd[key] = sd[key] * (1 - factor) + new_sd[key] * factor
        
        tqdm.write(f"{diff = :.2e}")
        progress_bar.update(1)
    
    progress_bar.close()
    
    sd_copy = load_file(lora_checkpoints_list[-1])
    final_diff = 0
    for key in sd.keys():
        final_diff += torch.nn.functional.mse_loss(sd[key], sd_copy[key]).item()
    print(f"{final_diff = :.2e}")
    
    output_path = os.path.splitext(lora_checkpoints_list[-1])[0] + f"_merged_last_{args.last_n}.safetensors"
    save_file(sd, output_path)
    print(f"saved to {output_path}")