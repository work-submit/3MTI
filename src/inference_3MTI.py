import os
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from model import Difix, load_ckpt_from_state_dict_noopt


def load_images(path):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, ext)))
    files = sorted(files)
    return files

def preprocess_like_training(input_path: str,
                             ref_path: str | None,
                             height: int,
                             width: int,
                             device: torch.device) -> torch.Tensor:
    
    inp = Image.open(input_path).convert("RGB")
    x_in = TF.to_tensor(inp)                           
    x_in = TF.resize(x_in, [128, 128])                 
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

    cond = cond.unsqueeze(0).to(device)               
    return cond


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to reference image or directory')
    parser.add_argument('--height', type=int, default=512, help='Network input height')
    parser.add_argument('--width', type=int, default=512, help='Network input width')
    parser.add_argument('--prompt', type=str, default="remove degradation", help='Base prompt')
    parser.add_argument('--model_name', type=str, default=None, help='HF model name (optional)')
    parser.add_argument('--model_path', type=str, default=None, help='Checkpoint path (.pkl)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='Save results as video')
    parser.add_argument("--mv_unet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=None,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    model = load_ckpt_from_state_dict_noopt(model, args.model_path)
    model.set_eval()
    print("Pretrained path:", args.model_path)

    if os.path.isdir(args.input_image):
        input_images = load_images(args.input_image)
    else:
        input_images = [args.input_image]

    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_images = load_images(args.ref_image)
        else:
            ref_images = [args.ref_image]
        assert len(input_images) == len(ref_images), "Input and reference counts must match."
    else:
        ref_images = [None] * len(input_images)

    prompt_dict = {}
    if os.path.exists("./prompt.txt"):
        with open("prompt.txt", "r") as f:
            for line in f:
                if ":" in line:
                    fname, extra = line.strip().split(":", 1)
                    prompt_dict[fname.strip()] = extra.strip()

    outputs = []
    for in_path, rf_path in tqdm(list(zip(input_images, ref_images)),
                                 desc="Processing images", total=len(input_images)):
        fname = os.path.basename(in_path)
        extra_prompt = prompt_dict.get(fname, "")
        final_prompt = args.prompt if extra_prompt == "" else f"{args.prompt}, {extra_prompt}"  # prompt combination
        print(f"{fname} full prompt: {final_prompt}")

        x = preprocess_like_training(in_path, rf_path, args.height, args.width, device)

        with torch.no_grad():
            y = model(x, prompt=final_prompt)       

        y_img = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu()
        out_pil = transforms.ToPILImage()(y_img)
        outputs.append(out_pil)

    if args.video:
        video_path = os.path.join(args.output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for out in tqdm(outputs, desc="Saving video"):
            writer.append_data(np.array(out))
        writer.close()
    else:
        for i, out in enumerate(tqdm(outputs, desc="Saving images")):
            out.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))

