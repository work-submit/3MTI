import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import glob
import sys
from multiprocessing import Process, set_start_method
from tqdm import tqdm
from PIL import Image
sys.path.append(os.getcwd())
from ram.models.ram_lora import ram as ram_lora
from ram import inference_ram as inference

# Image input
IMAGE_DIR = 'path_to_your_reference_image_folder'
# Prompt text output
OUTPUT_FILE_PATH = 'output_path_to/prompt.txt'

PRETRAINED_MODEL_PATH = 'path_to/ram_swin_large_14m.pth'
DAPE_CKPT_PATH = 'path_to/DAPE.pth'

gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
print(f"Available GPUs: {gpu_ids}")

ram_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_images(directory):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    image_lists = []
    for ext in image_extensions:
        image_lists.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    image_lists.sort()
    return image_lists

def load_model(device):
    print(f"[GPU {device.index if device.type == 'cuda' else 'CPU'}] Loading model...")
    model = ram_lora(
        pretrained=PRETRAINED_MODEL_PATH,
        image_size=384,
        vit='swin_l'
    )

    state_dict = torch.load(DAPE_CKPT_PATH, map_location='cpu')

    if 'params' in state_dict:
        lora_state_dict = state_dict['params']
    else:
        lora_state_dict = state_dict

    model.load_state_dict(lora_state_dict, strict=False)

    model = model.eval()
    model = model.to(device)
    print(f"[GPU {device.index if device.type == 'cuda' else 'CPU'}] Model loaded.")
    return model

def process_images(image_list, start_idx, end_idx, gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')
    
    model = load_model(device)
    
    local_image_list = image_list[start_idx:end_idx]
    print(f"[GPU {gpu_id}] Processing images from index {start_idx} to {end_idx-1} ({len(local_image_list)} images).")

    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out, torch.no_grad():
        for image_path in tqdm(local_image_list, desc=f'GPU {gpu_id} Processing'):
            try:
                basename = os.path.basename(image_path)
                image = Image.open(image_path).convert("RGB")
                image_tensor = ram_transforms(image).unsqueeze(0).to(device)

                captions = inference(image_tensor, model)

                tags = [tag.strip() for tag in captions[0].split(',') if tag.strip()]
                formatted_tags = ', '.join(tags)

                line = f"{basename}: {formatted_tags}\n"
                f_out.write(line)

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing image {image_path}: {e}")

    print(f"[GPU {gpu_id}] Finished processing.")

def main():
    image_lists = find_images(IMAGE_DIR)
    
    if not image_lists:
        print(f"Error: No images found in directory '{IMAGE_DIR}'")
        sys.exit(1)

    total_images = len(image_lists)
    print(f'Found {total_images} images in total.')

    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    if os.path.exists(OUTPUT_FILE_PATH):
        os.remove(OUTPUT_FILE_PATH)
        print(f"Cleared existing output file: {OUTPUT_FILE_PATH}")

    num_gpus = len(gpu_ids)
    images_per_gpu = total_images // num_gpus
    
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * images_per_gpu
        end_idx = total_images if i == num_gpus - 1 else start_idx + images_per_gpu
        
        p = Process(target=process_images, args=(image_lists, start_idx, end_idx, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll processing completed. Final results saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  
        
    main()