import os
import cv2
import numpy as np
import random

# ========== CONFIG ==========
gt_folder = "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/DL3DV/1K"  # 存放 GT 图像的文件夹
sys_folder = "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/DL3DV/1K_simulate_results"  # 存放 SYS 图像的文件夹
output_folder = "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/DL3DV/1K_ref"  # 输出结果文件夹

image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']


def process_pair_images(gt_img_path, sys_img_path, output_path):
    """处理一对图像并保存混合结果"""
    img_gt = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
    img_sys = cv2.imread(sys_img_path, cv2.IMREAD_COLOR)

    if img_gt is None or img_sys is None:
        print(f"跳过 {gt_img_path}: 图像读取失败")
        return

    # 预处理
    img_gt = cv2.resize(img_gt, (512, 512)).astype(np.float32) / 255.0
    img_sys = cv2.resize(img_sys, (512, 512)).astype(np.float32) / 255.0

    # 随机 alpha 值
    val = random.randint(30, 60)
    alpha = val / 100.0

    # 图像混合
    img_inter = alpha * img_gt + (1 - alpha) * img_sys
    img_inter = np.clip(img_inter, 0, 1)
    img_out = (img_inter * 255).astype(np.uint8)

    # 保存图像
    base_name = os.path.splitext(os.path.basename(gt_img_path))[0]
    save_name = f"{base_name}_alpha{val:02d}.png"
    save_path = os.path.join(output_path, save_name)
    cv2.imwrite(save_path, img_out)
    print(f" Saved {save_path}")


def process_sequence(gt_seq_path, sys_seq_path, output_seq_path):
    """处理一个子文件夹下的 images 文件夹中的所有图像"""
    # 进入 images 子文件夹
    gt_images_path = os.path.join(gt_seq_path, "images_4")
    sys_images_path = os.path.join(sys_seq_path, "images_4")

    if not os.path.exists(gt_images_path) or not os.path.exists(sys_images_path):
        print(f"跳过 {gt_seq_path}: 缺少 images_4 文件夹")
        return

    os.makedirs(output_seq_path, exist_ok=True)

    gt_images = sorted([f for f in os.listdir(gt_images_path) if os.path.splitext(f)[1].lower() in image_extensions])
    sys_images = sorted([f for f in os.listdir(sys_images_path) if os.path.splitext(f)[1].lower() in image_extensions])

    assert len(gt_images) == len(sys_images), f"{gt_images_path} 中图像数量不一致"

    for gt_name, sys_name in zip(gt_images, sys_images):
        gt_img_path = os.path.join(gt_images_path, gt_name)
        sys_img_path = os.path.join(sys_images_path, sys_name)
        process_pair_images(gt_img_path, sys_img_path, output_seq_path)



def batch_process(gt_root, sys_root, output_root):
    """递归处理所有子文件夹"""
    subfolders = [f for f in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, f))]

    for folder in subfolders:
        gt_seq_path = os.path.join(gt_root, folder)
        sys_seq_path = os.path.join(sys_root, folder)
        output_seq_path = os.path.join(output_root, folder)

        if not os.path.exists(sys_seq_path):
            print(f"跳过 {folder}: SYS 文件夹不存在")
            continue

        print(f"\n正在处理序列: {folder}")
        process_sequence(gt_seq_path, sys_seq_path, output_seq_path)


if __name__ == "__main__":
    batch_process(gt_folder, sys_folder, output_folder)
    print("\n所有图像处理完成！")