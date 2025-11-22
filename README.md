# 3MTI
3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion

## Setup

```bash
git clone https://github.com/work-submit/3MTI.git
cd 3MTI
conda create -n 3MTI python=3.10 pytorch=2.7.1 pytorch-cuda=11.8 -c pytorch -c nvidia
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

```
cd dataset
python create_json.py
```

## Semantic Extraction
Extract semantic information from the reference RGB images:
#### Step 1: Download the pretrained models
- Download the pretrained RAM (14M) model weight from [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth).
- Download the DAPE model weight from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link).
```
cd src
python create_json.py
```


