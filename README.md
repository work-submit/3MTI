# 3MTI
3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion

## Setup

```bash
git clone https://github.com/work-submit/3MTI.git
cd 3MTI
pip install -r requirements.txt
```

## Dataset preparation
Prepare your training set and test set in the following JSON format
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
