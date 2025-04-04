#!/bin/bash

# Check if experiment path argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_path> [name_suffix]"
    echo "Example: $0 output_f21_rank128_lr1e-5_shift3/2025-04-02_05-42-27 tile"
    exit 1
fi

# Set variables from arguments
EXPERIMENT_PATH="$1"
NAME_SUFFIX="${2:-}"  # Use empty string if not provided

# Extract experiment name from path (everything after output_ and before the date)
EXP_NAME=$(echo ${EXPERIMENT_PATH} | sed -E 's/output_([^/]+).*/\1/')

# Find the latest checkpoint in the experiment directory, excluding comfy versions
LATEST_CKPT=$(ls -v ../wan_data_spacepxl/${EXPERIMENT_PATH}/checkpoints/wan-lora-*.safetensors | grep -v "comfy" | tail -n 1)
CKPT_NAME=$(basename ${LATEST_CKPT} .safetensors)

echo "Converting checkpoint: ${LATEST_CKPT}"
echo "Experiment name: ${EXP_NAME}"

# Convert to ComfyUI format
python convert_lora_to_comfy.py --input_lora ${LATEST_CKPT}

# Move to checkpoint directory
cd ../wan_data_spacepxl/${EXPERIMENT_PATH}/checkpoints

# Rename with experiment details
if [ -z "${NAME_SUFFIX}" ]; then
    COMFY_NAME="wan-lora-${EXP_NAME}-${CKPT_NAME#wan-lora-}_comfy.safetensors"
else
    COMFY_NAME="wan-lora-${NAME_SUFFIX}-${EXP_NAME}-${CKPT_NAME#wan-lora-}_comfy.safetensors"
fi
echo "Comfy name: ${COMFY_NAME}"
mv ${CKPT_NAME}_comfy.safetensors ${COMFY_NAME}

# Copy to ComfyUI loras directory
cp ${COMFY_NAME} ../../../../ComfyUI/models/loras/

echo "Successfully converted and copied to ComfyUI:"
echo "../../../../ComfyUI/models/loras/${COMFY_NAME}"