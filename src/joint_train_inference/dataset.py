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
        # 应用
        for t in selected:
            img = t(img)
        return img

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, prompts_file=None):
        super().__init__()

        # 读取 JSON 文件
        with open(dataset_path, 'r') as f:
            json_data = json.load(f)[split]

        self.target_fusion_dir = json_data['target_fusion']  # 融合GT
        self.target_SR_dir = json_data['target_SR']          # 红外GT
        self.image_dir = json_data['image']                  # 输入RGB
        self.ref_dir = json_data['ref_image']                # 低分辨率红外
        self.prompt = json_data['prompt']                    # 正向提示词
        self.prompt_neg = json_data['prompt_neg']            # 负向提示词

        # 加载额外 prompt
        self.extra_prompts = {}
        if prompts_file is not None:
            with open(prompts_file, "r") as f:
                for line in f:
                    if ":" in line:
                        filename, prompt = line.strip().split(":", 1)
                        self.extra_prompts[filename.strip()] = prompt.strip()

        # rgb
        self.image_files = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        # 融合GT文件
        self.target_fusion_files = sorted([
            os.path.join(self.target_fusion_dir, f) for f in os.listdir(self.target_fusion_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        # 红外超分GT
        self.target_SR_files = sorted([
            os.path.join(self.target_SR_dir, f) for f in os.listdir(self.target_SR_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        # 低分辨率红外参考图
        self.ref_files = sorted([
            os.path.join(self.ref_dir, f) for f in os.listdir(self.ref_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        assert len(self.image_files) == len(self.target_fusion_files) == len(self.target_SR_files) == len(self.ref_files), \
            "输入、融合GT、红外SR GT、参考图像数量必须一致"

        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.color_trans = RandomSubsetColorJitter()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_path = self.image_files[idx]           # RGB 可见光输入
        ref_path = self.ref_files[idx]               # 低分辨率红外参考图

        # 🔥核心：70%正样本 / 30%负样本
        if random.random() < 0.7:
            target_fusion_path = self.target_fusion_files[idx]
            target_SR_path = self.target_SR_files[idx]
            current_prompt = self.prompt
            is_positive = True
        else:
            target_fusion_path = self.image_files[idx]
            target_SR_path = self.ref_files[idx]
            current_prompt = self.prompt_neg
            is_positive = False

        filename = os.path.basename(input_path)
        extra_prompt = self.extra_prompts.get(filename, "")
        final_prompt = f"{current_prompt}, {extra_prompt}" if extra_prompt != "" else current_prompt

        try:
            input_img = Image.open(input_path).convert("RGB")          # rgb
            target_fusion_img = Image.open(target_fusion_path).convert("RGB")  # 融合GT或rgb
            target_SR_img = Image.open(target_SR_path).convert("RGB")    # IR GT
            ref_img = Image.open(ref_path).convert("RGB")                # 低分辨率IR
        except Exception as e:
            print(f"Error loading images: {input_path}, {target_fusion_path}, {target_SR_path}, {ref_path}")
            return self.__getitem__((idx + 1) % len(self))

        input_tensor = F.to_tensor(input_img)
        input_tensor = F.resize(input_tensor, self.image_size, antialias=True)
        input_tensor = F.normalize(input_tensor, mean=[0.5]*3, std=[0.5]*3)

        ref_tensor = F.to_tensor(ref_img)
        ref_tensor = F.resize(ref_tensor, self.image_size, antialias=True)
        ref_tensor = F.normalize(ref_tensor, mean=[0.5]*3, std=[0.5]*3)  

        target_fusion_tensor = F.to_tensor(target_fusion_img)
        target_fusion_tensor = F.resize(target_fusion_tensor, self.image_size, antialias=True)
        target_fusion_tensor = F.normalize(target_fusion_tensor, mean=[0.5]*3, std=[0.5]*3)

        target_SR_tensor = F.to_tensor(target_SR_img)
        target_SR_tensor = F.resize(target_SR_tensor, self.image_size, antialias=True)
        target_SR_tensor = F.normalize(target_SR_tensor, mean=[0.5]*3, std=[0.5]*3)

        # x_src：rgb + 低分辨率红外
        x_src = torch.stack([input_tensor, ref_tensor], dim=0)   
        # x_tgt 融合GT(正或负通道) + 红外超分GT
        x_tgt = torch.stack([target_fusion_tensor, target_SR_tensor], dim=0)

        out = {
            "conditioning_pixel_values": x_src,  # [2, 3, H, W]
            "output_pixel_values": x_tgt,        # [2, 3, H, W]
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