import torch
import torch.nn.functional as F
import math

def RGB2YCrCb(rgb_image, with_CbCr=False):
    """
    将RGB转换为YCrCb颜色空间（符合人眼亮度感知）
    Args:
        rgb_image: [B, C, H, W]，值范围[0,1]
        with_CbCr: 是否返回拼接的CbCr通道
    Returns:
        Y: 亮度通道 [B,1,H,W]
        Cb/Cr: 色差通道 [B,1,H,W] 或 CbCr拼接 [B,2,H,W]
    """

    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128/255.0

    Y = Y.clamp(0.0, 1.0)
    Cb = Cb.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0)
    
    if with_CbCr:
        return Y, torch.cat([Cb, Cr], dim=1)
    return Y, Cb, Cr


def intensity_loss(fused_pre, fusion_gt):

    Y_fusison_pre, _, _ = RGB2YCrCb(fused_pre)
    Y_fusion_gt, _, _ = RGB2YCrCb(fusion_gt)

    return F.l1_loss(Y_fusison_pre, Y_fusion_gt, reduction='mean')

def color_loss(fused, rgb):
    """
    仅约束Cb/Cr色差通道
    """
    # 转换到YCrCb，提取Cb/Cr通道
    _, Cb_fused, Cr_fused = RGB2YCrCb(fused)
    _, Cb_rgb, Cr_rgb = RGB2YCrCb(rgb)

    loss_cb = F.l1_loss(Cb_fused, Cb_rgb)
    loss_cr = F.l1_loss(Cr_fused, Cr_rgb)
    
    return loss_cb + loss_cr