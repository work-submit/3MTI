python inference_BeyFusion.py \
--model_path "Path to pretrained model weight or your trained model weight" \
--input_image "Path to input high-resolution visible image" \
--ref_image "Path to degraded, calibration-free infrared image" \
--prompt "visible-infrared image fusion and infrared image super-resolution" \
--prompt_neg "The original visible image and low-resolution infrared image" \
--output_dir "Inference outputs save path" \
--cfg_scales_fu 1.1 \
--cfg_scales_sr 1.1 \
--mv_unet