import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import random
import argparse
from glob import glob
from tqdm.auto import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# borrowed and lightly modified from https://github.com/cseti007/Qwen2.5-VL-Video-Captioning
cseti_prompt = \
"""Please provide an analysis of this video by covering each of these aspects in your answer. Use only one paragraph. DO NOT separate your answer into topics.

1. **Main Content:**
    * What is the primary focus of the scene?
    * Who are the main characters visible?

2. **Object and Character Details:**
    * Don't refer to characters as 'individual', 'characters' and 'persons', instead always use their gender or refer to them with their gender.
    * Describe the appearance in detail
    * What notable objects are present?

3. **Actions and Movement:**
    * Describe ALL movements, no matter how subtle.
    * Specify the exact type of movement (walking, running, etc.).
    * Note the direction and speed of movements.

4. **Background Elements:**
    * Describe the setting and environment.
    * Note any environmental changes.

5. **Visual Style:**
    * Describe the lighting and color palette.
    * Note any special effects or visual treatments.
    * What is the overall style of the video? (e.g., realistic, animated, artistic, documentary)

6. **Camera Work:**
    * Describe EVERY camera angle change.
    * Note the distance from subjects (close-up, medium, wide shot).
    * Describe any camera movements (pan, tilt, zoom, handheld).

7. **Scene Transitions:**
    * If there are multiple shots included, describe what changes between each shot.
    * Note any changes in perspective or viewing angle.

Please be extremely specific and detailed in your description. If you notice any movement or changes, describe them explicitly."""

prompts = [
    "Describe this video.",
    "Write a short descriptive caption for this video by describing the subjects and actions.",
    cseti_prompt,
]

VIDEO_TYPES = [".mp4", ".mkv", ".mov", ".avi", ".webm"]


@torch.inference_mode()
def main(args):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    
    folders = glob(os.path.join(args.dataset, "*/"))
    for folder in tqdm(folders):
        video_files = []
        for ext in VIDEO_TYPES:
            video_files.extend(glob(os.path.join(folder, "**", "*" + ext), recursive=True))
        
        # for video_file in tqdm(video_files, leave=False):
        for video_file in video_files:
            caption_file = os.path.splitext(video_file)[0] + ".txt"
            if not os.path.exists(caption_file):
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_file,
                            # "max_pixels": 256 * 192,
                            # "max_pixels": 25_000,
                            # "min_pixels": 10_000,
                            # "fps": 0.5,
                            "min_pixels": 16 * 28 * 28,
                            "max_pixels": 32 * 28 * 28,
                            "min_frames": 4,
                            "max_frames": 64,
                        },
                        {
                            "type": "text",
                            "text": random.choice(prompts),
                        },
                    ],
                }]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(device)
                
                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                with open(caption_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                
                free, total = torch.cuda.mem_get_info(device)
                if free / 1024 ** 3 < 4:
                    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type = str,
        default = None,
        required = True,
        help = "Path to directory with video files to caption",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)