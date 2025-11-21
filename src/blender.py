import cv2
import numpy as np
import sys

# ========== CONFIG ==========
gt_path = sys.argv[1]              # 替换为你的GT图像路径
sys_path = sys.argv[2]     # 替换为你的系统模糊图像路径
# =============================

# Load images
img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
img_sys = cv2.imread(sys_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

img_gt = cv2.resize(img_gt,(512,512)).astype(np.float32) / 255.0

assert img_gt.shape == img_sys.shape, "GT and system images must be same size."

# Create window
cv2.namedWindow("Blender", cv2.WINDOW_NORMAL)

# Trackbar callback
def on_trackbar(val):
    alpha = val / 100.0
    img_inter = alpha * img_gt + (1 - alpha) * img_sys
    img_inter = np.clip(img_inter, 0, 1)
    img_show = (img_inter * 255).astype(np.uint8)
    cv2.imshow("Blender", img_show)

# Create trackbar
cv2.createTrackbar("Alpha (GT weight)", "Blender", 0, 100, on_trackbar)

# Initialize display
on_trackbar(0)

print("�� 使用说明:")
print(" - 拖动滑块实时查看中间效果")
print(" - 按 's' 保存当前结果")
print(" - 按 'q' 退出程序")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # Get current alpha
        val = cv2.getTrackbarPos("Alpha (GT weight)", "Blender")
        alpha = val / 100.0
        img_inter = alpha * img_gt + (1 - alpha) * img_sys
        img_inter = np.clip(img_inter, 0, 1)
        img_out = (img_inter * 255).astype(np.uint8)
        save_name = f"intermediate_alpha{val:02d}.png"
        cv2.imwrite(save_name, img_out)
        print(f"✅ Saved {save_name}")

cv2.destroyAllWindows()