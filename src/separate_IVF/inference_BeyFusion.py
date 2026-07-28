# multi cfg scale in latent space
import argparse
import os
from glob import glob

import torch
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import Difix, load_ckpt_from_state_dict_noopt


def load_images(path):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, ext)))
    return sorted(files)


def preprocess_like_training(
    input_path: str,
    ref_path: str | None,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return image conditioning in [B=1, V, C, H, W], normalized to [-1, 1]."""
    inp = Image.open(input_path).convert("RGB")
    x_in = TF.to_tensor(inp)
    x_in = TF.resize(x_in, [height, width])
    x_in = TF.normalize(x_in, mean=[0.5], std=[0.5])

    if ref_path is not None:
        ref = Image.open(ref_path).convert("RGB")
        x_ref = TF.to_tensor(ref)
        x_ref = TF.resize(x_ref, [height, width])
        x_ref = TF.normalize(x_ref, mean=[0.5], std=[0.5])
        cond = torch.stack([x_in, x_ref], dim=0)
    else:
        cond = x_in.unsqueeze(0)

    return cond.unsqueeze(0).to(device)


def encode_prompt(model: Difix, prompt: str, num_views: int) -> torch.Tensor:
    caption_tokens = model.tokenizer(
        prompt,
        max_length=model.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(model.text_encoder.device)
    caption_enc = model.text_encoder(caption_tokens)[0]
    return repeat(caption_enc, "b n c -> (b v) n c", v=num_views)


def predict_denoised_latent(
    model: Difix,
    z: torch.Tensor,
    caption_enc: torch.Tensor,
) -> torch.Tensor:
    model_pred = model.unet(z, model.timesteps, encoder_hidden_states=caption_enc).sample
    return model.sched.step(model_pred, model.timesteps, z, return_dict=True).prev_sample


def decode_latent(
    model: Difix,
    z_denoised: torch.Tensor,
    skip_acts: list[torch.Tensor],
    num_views: int,
) -> torch.Tensor:
    model.vae.decoder.incoming_skip_acts = skip_acts
    output_image = model.vae.decode(z_denoised / model.vae.config.scaling_factor).sample
    output_image = output_image.clamp(-1, 1)
    return rearrange(output_image, "(b v) c h w -> b v c h w", v=num_views)


def run_latent_cfg(
    model: Difix,
    x: torch.Tensor,
    pos_prompt: str,
    neg_prompt: str,
    cfg_scales: list[float],
) -> dict[float, torch.Tensor]:
    num_views = x.shape[1]
    x_flat = rearrange(x, "b v c h w -> (b v) c h w")

    # Encode once so positive and negative branches share the same latent sample.
    z = model.vae.encode(x_flat).latent_dist.sample() * model.vae.config.scaling_factor
    skip_acts = model.vae.encoder.current_down_blocks

    caption_pos = encode_prompt(model, pos_prompt, num_views)
    caption_neg = encode_prompt(model, neg_prompt, num_views)

    z_pos = predict_denoised_latent(model, z, caption_pos)
    z_neg = predict_denoised_latent(model, z, caption_neg)

    outputs = {}
    for cfg in cfg_scales:
        z_cfg = z_neg + cfg * (z_pos - z_neg)
        outputs[cfg] = decode_latent(model, z_cfg, skip_acts, num_views)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--ref_image", type=str, default=None, help="Path to reference image or directory")
    parser.add_argument("--height", type=int, default=512, help="Network input height")
    parser.add_argument("--width", type=int, default=512, help="Network input width")
    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt")
    parser.add_argument("--prompt_neg", type=str, required=True, help="Negative prompt")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output", help="ROOT output directory")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--mv_unet", action="store_true")
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        type=float,
        required=True,
        help="List of CFG scales, e.g. --cfg_scales 0.8 0.9 1.1 1.2",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=None,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    model = load_ckpt_from_state_dict_noopt(model, args.model_path)
    model.set_eval()
    print("Model loaded:", args.model_path)
    print("CFG scales:", args.cfg_scales)

    if os.path.isdir(args.input_image):
        input_images = load_images(args.input_image)
    else:
        input_images = [args.input_image]

    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_images = load_images(args.ref_image)
        else:
            ref_images = [args.ref_image]
        assert len(input_images) == len(ref_images)
    else:
        ref_images = [None] * len(input_images)

    prompt_dict = {}
    if os.path.exists("./video_02.txt"):
        with open("./video_02.txt") as f:
            for line in f:
                if ":" in line:
                    fname, extra = line.strip().split(":", 1)
                    prompt_dict[fname.strip()] = extra.strip()

    for in_path, rf_path in tqdm(zip(input_images, ref_images), total=len(input_images), desc="Processing"):
        fname = os.path.basename(in_path)
        extra = prompt_dict.get(fname, "")

        pos_prompt = f"{args.prompt}, {extra}" if extra else args.prompt
        neg_prompt = f"{args.prompt_neg}, {extra}" if extra else args.prompt_neg

        x = preprocess_like_training(in_path, rf_path, args.height, args.width, device)

        with torch.no_grad():
            cfg_outputs = run_latent_cfg(model, x, pos_prompt, neg_prompt, args.cfg_scales)

        for cfg, y in cfg_outputs.items():
            y_img = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu()
            out_pil = transforms.ToPILImage()(y_img)

            cfg_str = "cfg" + str(cfg).replace(".", "")
            out_subdir = os.path.join(args.output_dir, f"SR_12_{cfg_str}")
            os.makedirs(out_subdir, exist_ok=True)
            out_pil.save(os.path.join(out_subdir, fname))

    print("All results saved successfully!")
