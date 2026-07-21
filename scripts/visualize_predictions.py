# -*- coding: utf-8 -*-
"""
可视化 TransUNet 在 Synapse 上的预测结果
==========================================

把每个测试 case 的 CT / GT / Pred 三张 2D 切片叠在一张图上对比。
不需要 ITK-SNAP / 3D Slicer，纯 matplotlib 生成 PNG。

依赖：nibabel, matplotlib, numpy
    pip install nibabel matplotlib

使用：
    1) 确认 test.py 已经跑过，并使用了 --is_savenii（生成 .nii.gz）
    2) python scripts/visualize_predictions.py
    3) 在 ../visualizations/ 下查看 PNG

输出（在 ../visualizations/ 下）：
    case0035_compare.png   最佳 case (Dice 0.890)
    case0001_compare.png   中等 case (Dice 0.727)
    case0003_compare.png   困难 case (Dice 0.566)
    legend.png             9 类器官颜色对照
"""
import os
import sys

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示器环境必须
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# 路径配置（与你的实际目录对应，相对项目根）
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
WORKSPACE = os.path.dirname(PROJECT_ROOT)  # /media/ubuntu/Student/yinzhihan

PRED_DIR = os.path.join(WORKSPACE, "predictions",
                       "TU_Synapse224",
                       "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224")

# 把可视化输出镜像到 visualizations/<镜像 predictions 的子路径>/
#   predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/...
#   -> visualizations/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/...
_REL = os.path.relpath(PRED_DIR, os.path.join(WORKSPACE, "predictions"))
OUT_DIR = os.path.join(WORKSPACE, "visualizations", _REL)

# 9 类器官配色（对比度高、颜色盲友好）
ORGAN_COLORS = [
    "#000000",  # 0 背景
    "#E41A1C",  # 1 主动脉  red
    "#377EB8",  # 2 胆囊    blue
    "#4DAF4A",  # 3 左肾    green
    "#984EA3",  # 4 右肾    purple
    "#FF7F00",  # 5 肝      orange
    "#FFFF33",  # 6 胃      yellow
    "#A65628",  # 7 脾      brown
    "#F781BF",  # 8 胰      pink
]
ORGAN_NAMES = [
    "0 background",
    "1 aorta 主动脉",
    "2 gallbladder 胆囊",
    "3 left kidney 左肾",
    "4 right kidney 右肾",
    "5 liver 肝",
    "6 stomach 胃",
    "7 spleen 脾",
    "8 pancreas 胰",
]

CMAP = mcolors.ListedColormap(ORGAN_COLORS)
NORM = mcolors.Normalize(vmin=0, vmax=8)


def check_predictions_exist() -> bool:
    """检查预测目录是否有内容，避免空跑。"""
    if not os.path.isdir(PRED_DIR):
        print(f"✗ 预测目录不存在: {PRED_DIR}")
        print("  请先运行：python test.py --dataset Synapse --vit_name R50-ViT-B_16 "
              "--max_epochs 150 --batch_size 24 --is_savenii")
        return False
    nii_files = [f for f in os.listdir(PRED_DIR) if f.endswith(".nii.gz")]
    if not nii_files:
        print(f"✗ {PRED_DIR} 下没有 .nii.gz 文件")
        return False
    print(f"✓ 找到 {len(nii_files)} 个 .nii.gz 文件")
    return True


def parse_test_log() -> dict:
    """
    从 test.log 解析每个 case 的 Dice。
    期望格式: 'idx 0 case case0008 mean_dice 0.645844 mean_hd95 13.871876'
    返回: {case_name: mean_dice}

    查找顺序（找到第一个非空就返回）：
    1. TransUNet/test_log/  (test.py 默认输出)
    2. experiments/transunet_*/logs/test.log  (归档脚本可能已搬走)
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "test_log"),
        os.path.join(WORKSPACE, "experiments"),
    ]
    result = {}
    for log_dir in candidates:
        if not os.path.isdir(log_dir):
            continue
        # experiments/ 下要找的是 */logs/test.log；test_log/ 下是 *.txt
        txt_files = []
        if os.path.basename(log_dir) == "test_log":
            txt_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".txt")]
        else:
            for exp in os.listdir(log_dir):
                p = os.path.join(log_dir, exp, "logs", "test.log")
                if os.path.isfile(p):
                    txt_files.append(p)
                    break  # 找最新的一个就行
        for fpath in txt_files:
            with open(fpath, "r", errors="ignore") as f:
                for line in f:
                    if "mean_dice" in line and "case" in line:
                        parts = line.split()
                        try:
                            i = parts.index("case")
                            case_name = parts[i + 1]
                            j = parts.index("mean_dice")
                            dice = float(parts[j + 1])
                            result[case_name] = dice
                        except (ValueError, IndexError):
                            continue
        if result:
            return result
    return result


def pick_cases_by_dice(test_dice: dict) -> list:
    """
    从 test_dice 自动选 best / mid / worst 三个 case。

    返回: [(case_name, dice, tag), ...]
    """
    if not test_dice:
        return []

    sorted_cases = sorted(test_dice.items(), key=lambda kv: kv[1], reverse=True)
    n = len(sorted_cases)

    best = sorted_cases[0]
    worst = sorted_cases[-1]
    # 中位数（偶数个则取中间偏右的那个）
    mid = sorted_cases[n // 2]

    return [
        (best[0],  best[1],  "Best"),
        (mid[0],   mid[1],   "Mid"),
        (worst[0], worst[1], "Hard"),
    ]


def load_case(case: str):
    """读取一个 case 的 img / gt / pred。"""
    img_path  = os.path.join(PRED_DIR, f"{case}_img.nii.gz")
    gt_path   = os.path.join(PRED_DIR, f"{case}_gt.nii.gz")
    pred_path = os.path.join(PRED_DIR, f"{case}_pred.nii.gz")
    for p in [img_path, gt_path, pred_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺文件: {p}")
    img  = nib.load(img_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata().astype(int)
    pred = nib.load(pred_path).get_fdata().astype(int)
    return img, gt, pred


def pick_slices(gt: np.ndarray, n: int = 4) -> np.ndarray:
    """挑器官出现最多的 n 个切片。"""
    organ_per_slice = (gt > 0).sum(axis=(0, 1))
    return np.argsort(organ_per_slice)[-n:][::-1]


def plot_one_case(case: str, label: str, out_path: str) -> None:
    img, gt, pred = load_case(case)
    slices = pick_slices(gt, n=4)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i, s in enumerate(slices):
        axes[0, i].imshow(img[:, :, s], cmap='gray')
        axes[1, i].imshow(gt[:, :, s],   cmap=CMAP, norm=NORM, interpolation='nearest')
        axes[2, i].imshow(pred[:, :, s], cmap=CMAP, norm=NORM, interpolation='nearest')
        axes[0, i].set_title(f'slice {s}', fontsize=11)
        for ax in axes[:, i]:
            ax.axis('off')
            ax.set_aspect('equal')

    axes[0, 0].set_ylabel('CT',   fontsize=16, rotation=0, ha='right', va='center', labelpad=20)
    axes[1, 0].set_ylabel('GT',   fontsize=16, rotation=0, ha='right', va='center', labelpad=20)
    axes[2, 0].set_ylabel('Pred', fontsize=16, rotation=0, ha='right', va='center', labelpad=20)
    plt.suptitle(f"{case}  -  {label}", fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"✓ {out_path}")


def plot_legend(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, color) in enumerate(zip(ORGAN_NAMES, ORGAN_COLORS)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color, label=name))
    ax.legend(loc='center left', fontsize=13, bbox_to_anchor=(1, 0.5))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(ORGAN_NAMES) - 0.5)
    plt.title("9-class Organ Color Legend", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"✓ {out_path}")


def main() -> None:
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"workspace : {WORKSPACE}")
    print(f"预测目录  : {PRED_DIR}")
    print(f"输出目录  : {OUT_DIR}")
    print()

    if not check_predictions_exist():
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 从 test.log 自动解析每个 case 的 Dice
    test_dice = parse_test_log()
    if test_dice:
        print(f"✓ 从 test.log 解析到 {len(test_dice)} 个 case 的 Dice")
    else:
        print("⚠ 未找到 test.log，将跳过可视化（test.log 是自动选 case 的依据）")
        print("   请先跑：python test.py --dataset Synapse --vit_name R50-ViT-B_16 "
              "--max_epochs 150 --batch_size 24 --is_savenii")
        sys.exit(1)

    # 自动选 best / mid / worst 三个 case
    cases = pick_cases_by_dice(test_dice)
    print()
    print("选定的 3 个 case:")
    for case, dice, tag in cases:
        print(f"  {tag:5s}  {case}  Dice {dice:.3f}")

    print()
    for case, dice, tag in cases:
        label = f"{tag} (Dice {dice:.3f})"
        try:
            out = os.path.join(OUT_DIR, f"{case}_{tag.lower()}_compare.png")
            plot_one_case(case, label, out)
        except FileNotFoundError as e:
            print(f"⚠ 跳过 {case}: {e}")

    plot_legend(os.path.join(OUT_DIR, "_legend.png"))

    print("\n=== 完成 ===")
    print(f"所有 PNG 已生成在: {OUT_DIR}")
    print(f"对应训练权重路径 : {PRED_DIR}")
    print("用文件管理器打开 .png 即可查看效果。")


if __name__ == "__main__":
    main()