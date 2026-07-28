# (CVPR 2026) 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion

3M-TI: [Papers](https://arxiv.org/abs/2511.19117) | [Project Page](https://lab.xiaoyunyuan.net/index.html?project=3m-ti)

Authors: [Minchong Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=UX2ZtJcAAAAJ) | [Xiaoyun Yuan](https://xiaoyunyuan.net/) | [Junzhe Wan](https://scholar.google.com.hk/citations?view_op=list_works&hl=zh-CN&hl=zh-CN&user=QbbWdzEAAAAJ) | [Jianing Zhang]() | [Jun Zhang]()

## :rocket: Updates 

[2026-2-21] Our paper 3M-TI has been accepted by CVPR 2026. The code and dataset have been officially released.🎉🎉🎉

## 📱Mobile Thermal Imaging System
![3MTI](fig/fig_abs.png)

## 🔎Framework Overview
![3MTI](fig/fig_3m-ti.png)

## 📚Qualitative Results on Synthetic Dataset
![3MTI](fig/sota.png)

## 📷Qualitative Results on Mobile Imaging System
![3MTI](fig/val.png)

## ⚙️Setup
```bash
git clone https://github.com/work-submit/3MTI.git
cd 3MTI
conda create -n 3MTI python=3.10 -y
conda activate 3MTI
pip install -r requirements.txt
```

## Dataset Preparation
Prepare your training set and test set in the following JSON format:
```json
{
    "train": {
        "target_image": "path_to_target_high_resolution_thermal_image_folder",
        "image": "path_to_degraded_low_resolution_thermal_image_folder",
        "ref_image": "path_to_reference_high_resolution_RGB_image_folder",
        "prompt": "remove degradation"
    },
    "test": {
        "target_image": "path_to_target_high_resolution_thermal_image_folder",
        "image": "path_to_degraded_low_resolution_thermal_image_folder",
        "ref_image": "path_to_reference_high_resolution_RGB_image_folder",
        "prompt": "remove degradation"
    }
}
```
#### Step 1: Create a JSON file containing the dataset path.
```
cd dataset
python create_json.py
```

## Semantic Extraction (optional)
Extract semantic information from the reference RGB images:
#### Step 1: Download the pretrained models
- Download the pretrained RAM (14M) model weight from [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth).
- Download the DAPE model weight from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link).
- You can put these models into `3MTI/trained_model/`.
#### Step 2: Modify path
- Replace lines 16 to 21 of semantic_extract.py with your actual path.
```
IMAGE_DIR = 'path_to_your_reference_image_folder'
OUTPUT_FILE_PATH = 'output_path_to/prompt.txt'
PRETRAINED_MODEL_PATH = 'path_to/ram_swin_large_14m.pth'
DAPE_CKPT_PATH = 'path_to/DAPE.pth'
```
#### Step 3: Extraction
```
cd src
python semantic_extract.py
```
Your prompt.txt content should be formatted as follows:
```
00001.png: word1, word2, word3, ...
00002.png: word1, word2, word3, ...
...
XXXXX.png: word1, word2, word3, ...
```

## 🚀Inference
#### Step 1: Download the pretrained model
- Download the pretrained 3MTI model from [pretrained weights and data](https://pan.sjtu.edu.cn/web/share/7df7f0df32ac4cd4eecc243f5ff95483) or [HuggingFace](https://huggingface.co/MinchongChen1002/3M-TI/tree/main).
- You can put this model into `3MTI/trained_model/`.
#### Step 2: Modify path
- Replace lines 90 and 91 of inference_3MTI.py with your actual semantic prompt text path.
```
if os.path.exists("./prompt.txt"):
    with open("prompt.txt", "r") as f:
```
#### Step 3: Inference and save results
```bash
python inference_3MTI.py \
--model_path "path_to/trained_model/model.pkl" \
--input_image "path_to_your_low_resoluton_thermal_image_folder" \
--ref_image "path_to_your_high_resoluton_reference_RGB_image_folder" \
--prompt "remove degradation" \
--output_dir "path_to_inference_output_folder" \
--mv_unet
```

## 🌈 Train
#### Step 1: Modify path
- Replace lines 101 and 104 of train_3MTI.py with your actual semantic prompt text path.
```
dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer, prompts_file="path_to/training_prompt.txt")
dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer, prompts_file="path_to/test_prompt.txt")
```
#### Step 2: training
#### Single GPU
```bash
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
```
#### Multipe GPUs
```bash
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
```
## 🎒 Data open source
Our datasets used for training and validation are available at [pretrained weights and data](https://pan.sjtu.edu.cn/web/share/7df7f0df32ac4cd4eecc243f5ff95483) or [HuggingFace](https://huggingface.co/datasets/MinchongChen1002/3MTI_Datasets/tree/main).

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

📚 Note:

1. The proposed framework BeyondFusion supports both task-specific and joint training and inference. 
2. For the joint inference, our model simultaneously outputs infrared-visible image fusion (IVF) and infrared super-resolution (SR) results.
3. During the inference, the CFG scale is adjustable, where CFG_scale=1.1 is recommended in this work.

## Semantic Extraction (optional)
Extract semantic information from the high-resolution input/reference visible images (following the steps given above).

## Mode 1: Joint infrared SR and IVF
```bash
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
## Mode 1: Joint infrared SR and IVF
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
