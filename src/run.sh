## inference
# !/bin/bash
python inference_3MTI.py \
--model_path "path_to/trained_model/model.pkl" \
--input_image "path_to_your_low_resoluton_thermal_image_folder" \
--ref_image "path_to_your_high_resoluton_reference_RGB_image_folder" \
--prompt "remove degradation" \
--output_dir "path_to_inference_output_folder" \
--mv_unet
## training
# Single GPU
accelerate launch --mixed_precision=bf16 train_3MTI.py \
    --output_dir="path_to/saved_weights" \
    --dataset_path="path_to/your_dataset.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 2000 --viz_freq 10000 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet
# Multipe GPUs
export NUM_NODES=1
export NUM_GPUS=8
accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix.py \
    --output_dir="path_to/saved_weights" \
    --dataset_path="path_to/your_dataset.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 2000 --viz_freq 10000 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet