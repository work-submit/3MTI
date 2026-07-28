python inference_BeyFusion.py \
--model_path "Path to pretrained model weight or your trained model weight" \
--input_image "Path to input low-resolution infrared image" \
--ref_image "Path to calibration-free high-resolution visible image" \
--prompt "high-quality super-resolved infrared image" \
--prompt_neg "original degraded low-resolution infrared image" \
--output_dir "Inference outputs save path" \
--cfg_scales 1.1 \
--mv_unet