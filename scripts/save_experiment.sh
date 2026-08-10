#!/bin/bash
# 归档 TransUNet 训练/测试结果到带时间戳的目录（参数化版本）
#
# 用法:
#   bash scripts/save_experiment.sh --train-dir <TRAIN_DIR_NAME> [--note <备注>]
#
# 示例:
#   bash scripts/save_experiment.sh --train-dir TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224 --note baseline_150ep
#   bash scripts/save_experiment.sh --train-dir TU_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224 --note baseline_300ep
#
# TRAIN_DIR_NAME = model/TU_Synapse224/ 和 predictions/TU_Synapse224/ 下的子目录名。
# 指标自动从 test.log 解析，不再写死。

set -e

# ---------- 参数解析 ----------
TRAIN_DIR_NAME=""
NOTE="no_note"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-dir) TRAIN_DIR_NAME="$2"; shift 2 ;;
        --note)      NOTE="$2"; shift 2 ;;
        *)           NOTE="$1"; shift ;;   # 兼容旧用法：裸参数当作备注
    esac
done

if [ -z "$TRAIN_DIR_NAME" ]; then
    echo "ERROR: 缺少 --train-dir 参数"
    echo "用法: bash scripts/save_experiment.sh --train-dir <TRAIN_DIR_NAME> [--note <备注>]"
    echo "例如: bash scripts/save_experiment.sh --train-dir TU_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224 --note r50_300ep"
    exit 1
fi

DATE=$(date +%Y%m%d_%H%M%S)
NOTE_SAFE=$(echo "$NOTE" | tr ' /' '__')

# ---------- workspace 探测 ----------
if [ -d "/media/ubuntu/Student/yinzhihan" ]; then
    WORKSPACE="/media/ubuntu/Student/yinzhihan"
elif [ -d "/workspace" ]; then
    WORKSPACE="/workspace"
else
    echo "ERROR: 找不到 workspace 根目录"
    exit 1
fi

EXP_DIR="${WORKSPACE}/experiments/transunet_${DATE}_${NOTE_SAFE}"
SRC="${WORKSPACE}"
MODEL_DIR="${SRC}/model/TU_Synapse224/${TRAIN_DIR_NAME}"
PRED_DIR="${SRC}/predictions/TU_Synapse224/${TRAIN_DIR_NAME}"
VIZ_DIR="${SRC}/visualizations/TU_Synapse224/${TRAIN_DIR_NAME}"
TEST_LOG="${SRC}/TransUNet/test_log/test_log_TU_Synapse224/${TRAIN_DIR_NAME}.txt"

echo "=== Archiving to: ${EXP_DIR} ==="
mkdir -p "${EXP_DIR}"/{model,predictions,logs}

# ---------- 1. 模型权重：取目录里最大的 epoch_N.pth ----------
LAST_PTH=$(ls "${MODEL_DIR}"/epoch_*.pth 2>/dev/null | sort -V | tail -1 || true)
CHECKPOINT=""
if [ -n "$LAST_PTH" ]; then
    CHECKPOINT=$(basename "$LAST_PTH")
    cp "$LAST_PTH" "${EXP_DIR}/model/${CHECKPOINT}"
    echo "  ✓ model/${CHECKPOINT}"
else
    echo "  ⚠ model/ 为空: ${MODEL_DIR}"
fi

# ---------- 2. 预测结果（.nii.gz） ----------
if [ -d "$PRED_DIR" ]; then
    N_FILES=$(ls "$PRED_DIR"/*.nii.gz 2>/dev/null | wc -l)
    cp "$PRED_DIR"/*.nii.gz "${EXP_DIR}/predictions/" 2>/dev/null || true
    echo "  ✓ predictions/  (${N_FILES} 个 .nii.gz)"
fi

# ---------- 3. 训练日志 ----------
if [ -f "${MODEL_DIR}/log.txt" ]; then
    cp "${MODEL_DIR}/log.txt" "${EXP_DIR}/logs/train.log"
    echo "  ✓ logs/train.log"
fi

# ---------- 4. 测试日志（只拷匹配该 train-dir 的） ----------
if [ -f "$TEST_LOG" ]; then
    cp "$TEST_LOG" "${EXP_DIR}/logs/test.log"
    echo "  ✓ logs/test.log"
else
    # fallback：在 experiments/ 归档里找匹配的
    FOUND=$(find "${SRC}/experiments" -type f \( -name "test.log" -o -name "*.txt" \) \
            -path "*${TRAIN_DIR_NAME}*" 2>/dev/null | head -1)
    [ -n "$FOUND" ] && cp "$FOUND" "${EXP_DIR}/logs/test.log" && echo "  ✓ logs/test.log (fallback)"
fi

# ---------- 5. 可视化 PNG ----------
mkdir -p "${EXP_DIR}/visualizations"
if [ -d "$VIZ_DIR" ]; then
    cp "$VIZ_DIR"/*.png "${EXP_DIR}/visualizations/" 2>/dev/null || true
fi
N_VIZ=$(ls "${EXP_DIR}/visualizations/"*.png 2>/dev/null | wc -l)
[ "$N_VIZ" -gt 0 ] && echo "  ✓ visualizations/  (${N_VIZ} 个 PNG)"

# ---------- 6. 从目录名 + test.log 解析指标 ----------
EPOCHS=$(echo "$TRAIN_DIR_NAME" | grep -oP 'epo\K[0-9]+' || true)
[ -z "$EPOCHS" ] && EPOCHS="150"

TL="${EXP_DIR}/logs/test.log"
if [ -f "$TL" ]; then
    MEAN_DICE=$(grep -oP 'Testing performance in best val model: mean_dice : \K[0-9.]+' "$TL" | head -1)
    MEAN_HD95=$(grep -oP 'mean_hd95 : \K[0-9.]+' "$TL" | head -1)
fi
[ -z "$MEAN_DICE" ] && MEAN_DICE="N/A"
[ -z "$MEAN_HD95" ] && MEAN_HD95="N/A"

# 各器官：从 test.log 解析你的 Dice/HD95，论文值固定
ORGAN_CN=("主动脉" "胆囊" "左肾" "右肾" "肝" "胃" "脾" "胰")
PAPER_DICE=(0.875 0.698 0.841 0.867 0.950 0.832 0.918 0.776)
ROWS=""
SUM_YOUR=""
N_ROWS=0
if [ -f "$TL" ]; then
    for i in $(seq 1 8); do
        NAME="${i} (${ORGAN_CN[$((i-1))]})"
        PAPER="${PAPER_DICE[$((i-1))]}"
        LINE=$(grep "Mean class ${i} " "$TL" | tail -1)
        YOUR=$(echo "$LINE" | grep -oP 'mean_dice \K[0-9.]+' || true)
        HD=$(echo "$LINE" | grep -oP 'mean_hd95 \K[0-9.]+' || true)
        if [ -n "$YOUR" ]; then
            DIFF=$(awk -v a="$YOUR" -v b="$PAPER" 'BEGIN{printf "%.3f", a-b}')
            ROWS+=$(printf '| %s | %s | %s | %s | %s |\n' "$NAME" "$YOUR" "$PAPER" "$DIFF" "$HD")
            SUM_YOUR+=" $YOUR"
            N_ROWS=$((N_ROWS + 1))
        else
            ROWS+=$(printf '| %s | N/A | %s | N/A | N/A |\n' "$NAME" "$PAPER")
        fi
    done
fi
# 用 test.log 里的 mean_dice 直接作为平均行
[ -n "$MEAN_DICE" ] && MEAN_ROW=$(printf '| **平均** | **%s** | **0.842** | **N/A** | **%s** |' "$MEAN_DICE" "$MEAN_HD95")

echo "  ✓ 解析指标: mean_dice=${MEAN_DICE}, mean_hd95=${MEAN_HD95}, epochs=${EPOCHS}"

# ---------- 7. README.md ----------
cat > "${EXP_DIR}/README.md" << INNER_EOF
# TransUNet Synapse 训练记录

- **归档时间**: ${DATE}
- **备注**: ${NOTE}
- **训练目录**: ${TRAIN_DIR_NAME}
- **模型**: R50-ViT-B_16 (105.28M params)
- **数据集**: Synapse (BTCV) - 2211 训练切片 / 12 测试体积
- **预训练**: ImageNet21k R50+ViT-B_16
- **训练配置**: ${EPOCHS} epoch, batch_size=24, base_lr=0.01, SGD

## 测试结果 (12 cases)

| 指标 | 数值 |
|---|---|
| **平均 Dice** | ${MEAN_DICE} |
| **平均 HD95** | ${MEAN_HD95} mm |

## 各器官 Dice 与论文对比

| 器官 | 你的 Dice | TransUNet 论文 | 差距 | HD95 |
|---|---|---|---|---|
${ROWS}${MEAN_ROW} |

> 论文数值来自 Chen et al. 2021 (arXiv:2102.04306)。差距为负表示低于论文。
> 数值自动从 logs/test.log 解析，如需核对请直接查看该文件。

## 目录结构

- \`model/${CHECKPOINT}\` - 最终训练权重
- \`predictions/*.nii.gz\` - 12 cases × 3 类型（img / gt / pred）
- \`logs/train.log\` - 训练日志
- \`logs/test.log\` - 测试日志
- \`visualizations/*.png\` - CT/GT/Pred 可视化对比图

## 可视化

### 方式 1: PNG 对比图（推荐，无需任何软件）

直接打开 \`visualizations/caseXXXX_compare.png\`，每张图是 3×4 网格：
- 行 = CT / GT / Pred
- 列 = 4 张代表性切片
- 颜色对应器官：见 \`visualizations/_legend.png\`

### 方式 2: 原始 .nii.gz（用 ITK-SNAP / 3D Slicer）

打开 \`predictions/caseXXXX_img.nii.gz\` 作为主图像，
再加载 \`*_gt.nii.gz\` 和 \`*_pred.nii.gz\` 作为 segmentation 标签对比。
INNER_EOF
echo "  ✓ README.md"

# ---------- 8. config.json ----------
cat > "${EXP_DIR}/config.json" << INNER_EOF
{
  "model": "R50-ViT-B_16",
  "img_size": 224,
  "num_classes": 9,
  "max_epochs": ${EPOCHS},
  "batch_size": 24,
  "base_lr": 0.01,
  "optimizer": "SGD",
  "n_skip": 3,
  "vit_patches_size": 16,
  "seed": 1234,
  "train_dir": "${TRAIN_DIR_NAME}",
  "checkpoint": "${CHECKPOINT}",
  "test_dataset": "Synapse (BTCV)",
  "n_train_samples": 2211,
  "n_test_samples": 12,
  "test_metrics": {
    "mean_dice": ${MEAN_DICE},
    "mean_hd95": ${MEAN_HD95}
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
