import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import os
from torchvision import transforms
import random


class RandomSubsetColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p_each=0.5):
        self.transforms = [
            transforms.ColorJitter(brightness=brightness),
            transforms.ColorJitter(contrast=contrast),
            transforms.ColorJitter(saturation=saturation),
            transforms.ColorJitter(hue=hue)
        ]
        self.p_each = p_each

    def __call__(self, img):
        # 随机挑选部分变换
        selected = [t for t in self.transforms if random.random() < self.p_each]
        # 随机打乱顺序
        random.shuffle(selected)

        for t in selected:
            img = t(img)
        return img

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, prompts_file=None):
        super().__init__()

        # 读取 JSON 文件
        with open(dataset_path, 'r') as f:
            json_data = json.load(f)[split]

        self.image_dir = json_data['image']          
        self.target_dir = json_data['target_image']   # 正样本
        self.ref_dir = json_data.get('ref_image', None)

        self.prompt = json_data['prompt']
        self.prompt_neg = json_data['prompt_neg']

        # 加载额外 prompt
        self.extra_prompts = {}
        if prompts_file is not None:
            with open(prompts_file, "r") as f:
                for line in f:
                    if ":" in line:
                        filename, prompt = line.strip().split(":", 1)
                        self.extra_prompts[filename.strip()] = prompt.strip()

        # 文件列表
        self.image_files = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.target_files = sorted([
            os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if self.ref_dir:
            self.ref_files = sorted([
                os.path.join(self.ref_dir, f) for f in os.listdir(self.ref_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            self.ref_files = [None] * len(self.image_files)

        assert len(self.image_files) == len(self.target_files) == len(self.ref_files), \
            "输入、正负目标、参考图像数量必须一致"

        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.color_trans = RandomSubsetColorJitter()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_path = self.image_files[idx]
        ref_path = self.ref_files[idx]

        filename = os.path.basename(input_path)

        # 🔥核心：70%正样本 / 30%负样本
        if random.random() < 0.7:
            target_path = self.target_files[idx]
            current_prompt = self.prompt
            is_positive = True
        else:
            target_path = self.image_files[idx]
            current_prompt = self.prompt_neg
            is_positive = False

        # extra prompt
        extra_prompt = self.extra_prompts.get(filename, "")
        final_prompt = f"{current_prompt}, {extra_prompt}" if extra_prompt != "" else current_prompt

        # Load images
        try:
            input_img = Image.open(input_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            ref_img = Image.open(ref_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {input_path}, {target_path}, {ref_path}")
            return self.__getitem__((idx + 1) % len(self))

        # --------------------------
        # Input RGB
        # --------------------------
        input_tensor = F.to_tensor(input_img)
        input_tensor = F.resize(input_tensor, self.image_size, antialias=True)
        input_tensor = F.normalize(input_tensor, mean=[0.5]*3, std=[0.5]*3)

        # --------------------------
        # Target（正 or 负）
        # --------------------------
        target_tensor = F.to_tensor(target_img)
        target_tensor = F.resize(target_tensor, self.image_size, antialias=True)
        target_tensor = F.normalize(target_tensor, mean=[0.5]*3, std=[0.5]*3)

        # --------------------------
        # Reference IR
        # --------------------------
        ref_tensor = F.to_tensor(ref_img)
        ref_tensor = F.resize(ref_tensor, self.image_size, antialias=True)
        ref_tensor = F.normalize(ref_tensor, mean=[0.5], std=[0.5])

        # --------------------------
        # Stack
        # --------------------------
        x_src = torch.stack([input_tensor, ref_tensor], dim=0)
        x_tgt = torch.stack([target_tensor, ref_tensor], dim=0)

        out = {
            "conditioning_pixel_values": x_src,
            "output_pixel_values": x_tgt,
            "caption": final_prompt,
            "filename": filename,
            "is_positive": is_positive
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                final_prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out