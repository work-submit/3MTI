# (3M-TI Extension) BeyondFusion: Self-Aligned Latent Diffusion for Calibration-Free Infrared Super-Resolution and Infrared-Visible Fusion

BeyondFusion: [arXiv](https://arxiv.org/abs/2607.24110) | [Project Page](https://xiaoyunyuan.net/index.html?project=beyondfusion)

Authors: [Minchong Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=UX2ZtJcAAAAJ) | [Xiaoyun Yuan](https://xiaoyunyuan.net/) | [Minyu Cao]() | [Jianing Zhang]() | [Jun Zhang]() | [Shuyang Liu]() | [Xiaokang Yang](https://icne.sjtu.edu.cn/info/1064/1078.htm)

## 🔎Framework Overview
![BeyondFusion](fig/BeyondFusion_framework.png)

## 📷Qualitative Results on Mobile Imaging System
<p align="center">
<img src="fig/Fusion_val.png" width="600" alt="Fusion validation results">
</p>

## 📌Downstream Pedestrian Detection
<p align="center">
<img src="fig/Detection.png" width="600" alt="Pedestrian detection results">
</p>

Our datasets used for training and validation are available at [pretrained weights and data](https://pan.sjtu.edu.cn/web/share/365c881396aeeafd63269b1ca4ca1f6f).

## 🚀Inference
Download BeyondFusion model weights at [pretrained weights and data](https://pan.sjtu.edu.cn/web/share/365c881396aeeafd63269b1ca4ca1f6f) or [huggingface](https://huggingface.co/MinchongChen1002/BeyondFusion/tree/main).

📚 Note:

1. The proposed framework BeyondFusion supports both task-specific and joint training and inference. 
2. For the joint inference, our model simultaneously outputs infrared-visible image fusion (IVF) and infrared super-resolution (SR) results.
3. During the inference, the CFG scale is adjustable, where CFG_scale=1.1 is recommended in this work.

## Semantic Extraction (optional)
Extract semantic information from the high-resolution input/reference visible images (following the steps given in 3M-TI-main branch).

## Mode 1: Joint Infrared SR and IVF
```bash
conda activate 3MTI
cd 3MTI-extension
cd src
cd joint_train_inference
```

## Run Inference
```bash
python inference_BeyFusion.py \
--model_path "Path to pretrained model weight (BeyFusion_joint_SR_IVF.pkl) or your trained model weight" \
--input_image "Path to input high-resolution visible image" \
--ref_image "Path to degraded, calibration-free infrared image" \
--prompt "visible-infrared image fusion and infrared image super-resolution" \
--prompt_neg "The original visible image and low-resolution infrared image" \
--output_dir "Inference outputs save path" \
--cfg_scales_fu 1.1 \
--cfg_scales_sr 1.1 \
--mv_unet
```

## Mode 2: Single Infrared SR
```bash
cd 3MTI-extension
cd src
cd separate_infrared_SR
```

## Run Inference
```bash
python inference_BeyFusion.py \
--model_path "Path to pretrained model weight (BeyFusion_infrared_SR.pkl) or your trained model weight" \
--input_image "Path to input low-resolution infrared image" \
--ref_image "Path to calibration-free high-resolution visible image" \
--prompt "high-quality super-resolved infrared image" \
--prompt_neg "original degraded low-resolution infrared image" \
--output_dir "Inference outputs save path" \
--cfg_scales 1.1 \
--mv_unet
```

## Mode 3: Single IVF
```bash
cd 3MTI-extension
cd src
cd separate_IVF
```

## Run Inference
```bash
python inference_BeyFusion.py \
--model_path "Path to pretrained model weight (BeyFusion_IVF.pkl) or your trained model weight" \
--input_image "Path to input high-resolution visible image" \
--ref_image "Path to degraded, calibration-free infrared image" \
--prompt "a high-quality fused image, salient objects clearly highlighted, clear structure and rich details, complementary information from visible and thermal modalities" \
--prompt_neg "an RGB-only image lacking infrared information, neglecting thermal cues, single-modality visible image" \
--output_dir "Inference outputs save path" \
--cfg_scales 1.1 \
--mv_unet
```

## 🌈 Train
## Mode 1: Joint Infrared SR and IVF
```bash
cd 3MTI-extension
cd src
cd joint_train_inference
```

## Dataset Preparation
Fill the joint_dataset.json in the data file:
```json
{
    "train": {
        "target_fusion": "Path to target fusion image",
        "target_SR": "Path to target high-resolution infrared image",
        "image": "Path to input high-resolution visible image",
        "ref_image": "Path to degraded, calibration-free infrared image",
        "prompt": "visible-infrared image fusion and infrared image super-resolution",
        "prompt_neg": "The original visible image and low-resolution infrared image"
    },
    "test": {
        "target_fusion": "Path to target fusion image",
        "target_SR": "Path to target high-resolution infrared image",
        "image": "Path to input high-resolution visible image",
        "ref_image": "Path to degraded, calibration-free infrared image",
        "prompt": "visible-infrared image fusion and infrared image super-resolution",
        "prompt_neg": "The original visible image and low-resolution infrared image"
    }
}
```

## Run Training
```
accelerate launch --mixed_precision=bf16 train_BeyFusion.py \
    --output_dir="Model weights save path" \
    --dataset_path="./data/joint_dataset.json" \
    --max_train_steps 16000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 --eval_freq 2000 --viz_freq 10000 \
    --lambda_int_pos 1.0 \
    --lambda_int_neg 0.0 \
    --lambda_color_pos 1.0 \
    --lambda_color_neg 0.0 \
    --lambda_l2 2.0 \
    --lambda_l2_sr 10.0 \
    --lambda_lpips 1.0 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet
```

## Mode 2: Single Infrared SR
```bash
cd 3MTI-extension
cd src
cd separate_infrared_SR
```

## Dataset Preparation
Fill the Infrared_SR_dataset.json in the data file:
```json
{
    "train": {
        "target_image": "Path to target high-resolution infrared image",
        "image": "Path to input low-resolution infrared image",
        "ref_image": "Path to calibration-free high-resolution visible image",
        "prompt": "high-quality super-resolved infrared image",
        "prompt_neg": "original degraded low-resolution infrared image"
    },
    "test": {
        "target_image": "Path to target high-resolution infrared image",
        "image": "Path to input low-resolution infrared image",
        "ref_image": "Path to calibration-free high-resolution visible image",
        "prompt": "high-quality super-resolved infrared image",
        "prompt_neg": "original degraded low-resolution infrared image"
    }
}
```

## Run Training
```
accelerate launch --mixed_precision=bf16 train_BeyFusion.py \
    --output_dir="Model weights save path" \
    --dataset_path="./data/Infrared_SR_dataset.json" \
    --max_train_steps 12000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 --eval_freq 2000 --viz_freq 10000 \
    --lambda_l2 10.0 \
    --lambda_lpips 1.0 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet
```

## Mode 3: Single IVF
```bash
cd 3MTI-extension
cd src
cd separate_IVF
```

## Dataset Preparation
Fill the IVF_dataset.json in the data file:
```json
{
    "train": {
        "target_image": "Path to target fusion image",
        "image": "Path to input high-resolution visible image",
        "ref_image": "Path to degraded, calibration-free infrared image",
        "prompt": "a high-quality fused image, salient objects clearly highlighted, clear structure and rich details, complementary information from visible and thermal modalities",
        "prompt_neg": "an RGB-only image lacking infrared information, neglecting thermal cues, single-modality visible image"
    },
    "test": {
        "target_image": "Path to target fusion image",
        "image": "Path to input high-resolution visible image",
        "ref_image": "Path to degraded, calibration-free infrared image",
        "prompt": "a high-quality fused image, salient objects clearly highlighted, clear structure and rich details, complementary information from visible and thermal modalities",
        "prompt_neg": "an RGB-only image lacking infrared information, neglecting thermal cues, single-modality visible image"
    }
}
```

## Run Training
```
accelerate launch --mixed_precision=bf16 train_BeyFusion.py \
    --output_dir="Model weights save path" \
    --dataset_path="./data/IVF_dataset.json" \
    --max_train_steps 16000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 --eval_freq 2000 --viz_freq 10000 \
    --lambda_int_pos 1.0 \
    --lambda_int_neg 0.0 \
    --lambda_color_pos 1.0 \
    --lambda_color_neg 0.0 \
    --lambda_l2 2.0 \
    --lambda_lpips 1.0 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet
```
