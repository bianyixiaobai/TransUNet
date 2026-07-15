# -*- coding: utf-8 -*-
"""
Smoke test for TransUNet (R50-ViT-B_16).

目的（4 步全验证）：
    1. Python 环境 / PyTorch / CUDA 链路通畅；
    2. 数据 / 权重路径对得上；
    3. 网络结构能正常 build、load_from、forward、backward；
    4. 真实 npz 权重能成功 load 进模型（不只是占位）。

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
    """构建 TransUNet 并返回 model 与 config（调用方决定是否 load_from）。"""
    config = CONFIGS_ViT_seg[vit_name]
    config.n_classes = num_classes
    config.n_skip = 3
    # R50 变体需要这个 grid
    if vit_name.find("R50") != -1:
        config.patches.grid = (int(img_size / 16), int(img_size / 16))
    model = ViT_seg(config, img_size=img_size, num_classes=config.n_classes)
    return model, config


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

    # ---- 1. 检查数据 / 权重路径 ----
    print("\n[1/5] Check data / weights paths ...")
    train_npz_dir = "../data/Synapse/train_npz"
    pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"

    n_train = len([f for f in os.listdir(train_npz_dir)
                   if f.endswith(".npz")]) if os.path.isdir(train_npz_dir) else 0
    print(f"      train_npz dir  : {train_npz_dir}")
    print(f"      -> exists={os.path.isdir(train_npz_dir)}, files={n_train}")
    assert os.path.isdir(train_npz_dir), f"训练数据目录不存在: {train_npz_dir}"
    assert n_train > 1000, f"训练数据数量异常: {n_train}（预期 > 1000）"

    print(f"      weights path   : {pretrained_path}")
    print(f"      -> exists={os.path.isfile(pretrained_path)}, "
          f"size={os.path.getsize(pretrained_path)/1024**2:.1f} MB" if os.path.isfile(pretrained_path) else "MISSING")
    assert os.path.isfile(pretrained_path), f"预训练权重不存在: {pretrained_path}"

    # 顺便看一眼 npz 的 keys（应该 ~217 个 R50+ViT-B/16 keys）
    npz = np.load(pretrained_path)
    keys = list(npz.keys())
    print(f"      npz keys count : {len(keys)}")
    print(f"      first 5 keys   : {keys[:5]}")
    assert len(keys) > 100, f"npz keys 数量异常: {len(keys)}（预期 ~217）"

    # ---- 2. 构建模型 ----
    img_size, num_classes, batch_size = 224, 9, 2
    print(f"\n[2/5] Build model: R50-ViT-B_16, img_size={img_size}, num_classes={num_classes}")
    model, config = build_model(img_size=img_size, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    # 顺手确认 config 关键字段（消除 unused 警告 + 顺便 sanity check）
    print(f"      config.pretrained_path = {config.pretrained_path}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      total params     : {n_params/1e6:.2f} M")
    print(f"      trainable params : {n_trainable/1e6:.2f} M")

    # ---- 3. 加载预训练权重（关键！） ----
    print("\n[3/5] Load pretrained weights (R50+ViT-B_16) ...")
    # train.py 里的写法：np.load + model.load_from
    # load_from 内部会过滤掉不匹配的 key，并打印若干 Loading、Skip 提示
    try:
        model.load_from(np.load(pretrained_path))
        print("      load_from OK ✓")
    except Exception as e:
        print(f"      load_from FAILED: {e}")
        raise

    # 把模型搬上 GPU
    model = model.to(device)

    # ---- 4. Forward ----
    print(f"\n[4/5] Forward pass (batch={batch_size}) ...")
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, num_classes, (batch_size, img_size, img_size), device=device)

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

    # ---- 5. Backward（用 0.5*CE + 0.5*Dice 复刻 trainer.py 的损失） ----
    print("\n[5/5] Backward pass ...")
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

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed! backward chain broken."

    print("\n" + "=" * 60)
    print("[OK] 全流程通过 ✓")
    print("     ✓ 环境与 CUDA")
    print("     ✓ 数据与权重路径")
    print("     ✓ 模型构建")
    print("     ✓ 预训练权重加载")
    print("     ✓ 前向 + 反向")
    print()
    print("下一步：在服务器上跑 python train.py 即可开始训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()