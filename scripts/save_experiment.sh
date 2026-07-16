#!/bin/bash
# 归档当前 TransUNet 训练/测试结果到带时间戳的目录
# 用法: bash scripts/save_experiment.sh [备注]

set -e
NOTE="${1:-no_note}"
DATE=$(date +%Y%m%d_%H%M%S)

# 自动探测 workspace 根目录
if [ -d "/media/ubuntu/Student/yinzhihan" ]; then
    WORKSPACE="/media/ubuntu/Student/yinzhihan"
elif [ -d "/workspace" ]; then
    WORKSPACE="/workspace"
else
    echo "ERROR: 找不到 workspace 根目录"
    exit 1
fi

EXP_DIR="${WORKSPACE}/experiments/transunet_${DATE}_${NOTE}"
SRC="${WORKSPACE}"
TRAIN_DIR_NAME="TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224"
PRED_DIR_NAME="TU_Synapse224/${TRAIN_DIR_NAME}"

echo "=== Archiving to: ${EXP_DIR} ==="
mkdir -p "${EXP_DIR}"/{model,predictions,logs}

# 1. 模型权重（最终 epoch_149）
if [ -f "${SRC}/model/TU_Synapse224/${TRAIN_DIR_NAME}/epoch_149.pth" ]; then
    cp "${SRC}/model/TU_Synapse224/${TRAIN_DIR_NAME}/epoch_149.pth" \
       "${EXP_DIR}/model/epoch_149.pth"
    echo "  ✓ model/epoch_149.pth"
fi

# 2. 预测结果（.nii.gz）
if [ -d "${SRC}/predictions/${PRED_DIR_NAME}" ]; then
    N_FILES=$(ls "${SRC}/predictions/${PRED_DIR_NAME}"/*.nii.gz 2>/dev/null | wc -l)
    cp "${SRC}/predictions/${PRED_DIR_NAME}"/*.nii.gz "${EXP_DIR}/predictions/" 2>/dev/null || true
    echo "  ✓ predictions/  (${N_FILES} 个 .nii.gz)"
fi

# 3. 训练日志
if [ -f "${SRC}/model/TU_Synapse224/${TRAIN_DIR_NAME}/log.txt" ]; then
    cp "${SRC}/model/TU_Synapse224/${TRAIN_DIR_NAME}/log.txt" \
       "${EXP_DIR}/logs/train.log"
    echo "  ✓ logs/train.log"
fi

# 4. 测试日志
if [ -d "${SRC}/TransUNet/test_log" ]; then
    find "${SRC}/TransUNet/test_log" -name "*.txt" -exec cp {} "${EXP_DIR}/logs/test.log" \; 2>/dev/null
    echo "  ✓ logs/test.log"
fi

# 5. 写一个 README 总结
cat > "${EXP_DIR}/README.md" << INNER_EOF
# TransUNet Synapse 训练记录

- **归档时间**: ${DATE}
- **备注**: ${NOTE}
- **模型**: R50-ViT-B_16 (105.28M params)
- **数据集**: Synapse (BTCV) - 2211 训练切片 / 12 测试体积
- **预训练**: ImageNet21k R50+ViT-B_16
- **训练配置**: 150 epoch, batch_size=24, base_lr=0.01, SGD
- **总训练时长**: ~41 分钟 (RTX 4090)

## 测试结果 (12 cases)

| 指标 | 数值 |
|---|---|
| **平均 Dice** | 0.7662 |
| **平均 HD95** | 30.06 mm |

## 各器官 Dice

| 器官 | Dice | HD95 |
|---|---|---|
| 1 (aorta 主动脉) | 0.867 | 8.53 |
| 2 (gallbladder 胆囊) | 0.620 | 26.81 |
| 3 (left kidney 左肾) | 0.802 | 51.08 |
| 4 (right kidney 右肾) | 0.730 | 60.95 |
| 5 (liver 肝) | 0.943 | 22.43 |
| 6 (stomach 胃) | 0.539 | 17.91 |
| 7 (spleen 脾) | 0.864 | 36.08 |
| 8 (pancreas 胰) | 0.764 | 16.74 |

## 目录结构

- \`model/epoch_149.pth\` - 最终训练权重 (440 MB)
- \`predictions/*.nii.gz\` - 36 个 .nii.gz（12 cases × 3 类型）
  - \`*_img.nii.gz\` - 原始 CT
  - \`*_gt.nii.gz\` - 真实标注
  - \`*_pred.nii.gz\` - 模型预测
- \`logs/train.log\` - 训练日志
- \`logs/test.log\` - 测试日志

## 可视化

用 ITK-SNAP 打开 \`predictions/caseXXXX_img.nii.gz\` 作为主图像，
然后加载 \`*_gt.nii.gz\` 和 \`*_pred.nii.gz\` 作为 segmentation 标签对比。
INNER_EOF
echo "  ✓ README.md"

# 6. 写配置信息快照
cat > "${EXP_DIR}/config.json" << INNER_EOF
{
  "model": "R50-ViT-B_16",
  "img_size": 224,
  "num_classes": 9,
  "max_epochs": 150,
  "batch_size": 24,
  "base_lr": 0.01,
  "optimizer": "SGD",
  "n_skip": 3,
  "vit_patches_size": 16,
  "seed": 1234,
  "checkpoint": "epoch_149.pth",
  "test_dataset": "Synapse (BTCV)",
  "n_train_samples": 2211,
  "n_test_samples": 12,
  "test_metrics": {
    "mean_dice": 0.7662,
    "mean_hd95": 30.06
  }
}
INNER_EOF
echo "  ✓ config.json"

echo ""
echo "=== Done ==="
echo "存档位置: ${EXP_DIR}"
du -sh "${EXP_DIR}"/* 2>/dev/null
echo ""
echo "总大小:"
du -sh "${EXP_DIR}"
