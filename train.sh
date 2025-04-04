python train_wan_lora_wlt_data.py \
    --dataset /mnt/filestore/users/bmild/wan_data_spacepxl/input \
    --output_dir /mnt/filestore/users/bmild/wan_data_spacepxl/output_depth_f21_rank128_lr1e-5_shift3 \
    --control_lora \
    --control_preprocess depth \
    --lora_rank 128 \
    --max_frame_stride 1 \
    --max_train_steps 100000 \
    --checkpointing_steps 1000 \
    --val_samples 10 \
    --learning_rate 1e-5 \
    --shift 3.0 \

    # --token_limit 10000 \
    # --max_frame_stride 4 \
    # --base_res 624 \
