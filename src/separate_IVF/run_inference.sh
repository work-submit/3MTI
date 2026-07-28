python inference_BeyFusion.py \
--model_path "Path to pretrained model weight or your trained model weight" \
--input_image "Path to input high-resolution visible image" \
--ref_image "Path to degraded, calibration-free infrared image" \
--prompt "a high-quality fused image, salient objects clearly highlighted, clear structure and rich details, complementary information from visible and thermal modalities" \
--prompt_neg "an RGB-only image lacking infrared information, neglecting thermal cues, single-modality visible image" \
--output_dir "Inference outputs save path" \
--cfg_scales 1.1 \
--mv_unet