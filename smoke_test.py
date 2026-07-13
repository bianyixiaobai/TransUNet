# -*- coding: utf-8 -*-
"""
Smoke test for TransUNet (R50-ViT-B_16) on a single synthetic image.

目的：
    1. 验证 Python 环境 / PyTorch / CUDA 链路通畅；
    2. 验证网络结构能正常 build、forward、backward；
    3. 不依赖任何预训练权重和数据集，开箱即用。

用法（在项目根目录，且已 conda activate transunet）：
    python smoke_test.py
"""
import os
import sys
import time

import numpy as np
import torch

# 让脚本能 import 到 networks/ 下的包
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg    # noqa: E402


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(img_size: int = 224, num_classes: int = 9, vit_name: str = "R50-ViT-B_16"):
    """构建一个不加载预训练权重的 TransUNet。"""
    config = CONFIGS_ViT_seg[vit_name]
    config.n_classes = num_classes
    config.n_skip = 3
    # R50 变体需要这个 grid
    if vit_name.find("R50") != -1:
        config.patches.grid = (int(img_size / 16), int(img_size / 16))
    model = ViT_seg(config, img_size=img_size, num_classes=config.n_classes)
    return model


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 基本环境信息 ----
    print("=" * 60)
    print(f"torch        : {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device       : {torch.cuda.get_device_name(0)}")
        print(f"capability   : {torch.cuda.get_device_capability(0)}")
    print("=" * 60)

    # ---- 1. 构建模型 ----
    img_size, num_classes, batch_size = 224, 9, 2
    print(f"\n[1/4] Build model: R50-ViT-B_16, img_size={img_size}, num_classes={num_classes}")
    model = build_model(img_size=img_size, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      total params     : {n_params/1e6:.2f} M")
    print(f"      trainable params : {n_trainable/1e6:.2f} M")

    # ---- 2. 构造随机输入 ----
    print(f"\n[2/4] Make random input: x.shape = ({batch_size}, 3, {img_size}, {img_size})")
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, num_classes, (batch_size, img_size, img_size), device=device)

    # ---- 3. Forward ----
    print("\n[3/4] Forward pass ...")
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_fwd = time.time() - t0
    print(f"      logits.shape = {tuple(logits.shape)}  (expect: ({batch_size}, {num_classes}, {img_size}, {img_size}))")
    assert logits.shape == (batch_size, num_classes, img_size, img_size), \
        f"Output shape mismatch! got {tuple(logits.shape)}"
    print(f"      forward time : {dt_fwd*1000:.1f} ms")
    if device.type == "cuda":
        print(f"      GPU mem alloc : {torch.cuda.memory_allocated()/1024**2:.1f} MiB")
        print(f"      GPU mem peak  : {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")

    # ---- 4. Backward（用 0.5*CE + 0.5*Dice 复刻 trainer.py 的损失） ----
    print("\n[4/4] Backward pass ...")
    from utils import DiceLoss
    from torch.nn.modules.loss import CrossEntropyLoss

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    optimizer.zero_grad()
    loss_ce = ce_loss(logits, y.long())
    loss_dice = dice_loss(logits, y, softmax=True)
    loss = 0.5 * loss_ce + 0.5 * loss_dice
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_bwd = time.time() - t0
    print(f"      loss_ce   = {loss_ce.item():.4f}")
    print(f"      loss_dice = {loss_dice.item():.4f}")
    print(f"      loss      = {loss.item():.4f}")
    print(f"      backward+step time : {dt_bwd*1000:.1f} ms")

    # 简单 sanity：拿一个参数，看它有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed! backward chain broken."

    print("\n[OK] Smoke test passed. 网络结构 / GPU 前向反向 均正常。")
    print("     下一步：放好预训练权重和数据集后即可 python train.py ...")


if __name__ == "__main__":
    main()
