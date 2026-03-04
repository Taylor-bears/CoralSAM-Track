#!/bin/bash
# ============================================================
# CoralSAM-Track 完整流水线脚本（Linux 远程服务器）
# 功能：环境检查 → 基线推理 → 漂移纠偏推理 → 评测对比
# ============================================================

# ────────────────────────────────────────────────────────────
# ★ 基本路径配置（按照服务器实际路径修改）
# ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 脚本所在目录即项目根目录
PYTHON="/data1/baiyang/anaconda/envs/coralsam/bin/python"      # ← 按服务器 conda 路径修改
CONFIG="configs/default.yaml"

# ────────────────────────────────────────────────────────────
# ★ CUDA 设备（填写服务器显卡编号，多卡只取 id=0 的那张）
# ────────────────────────────────────────────────────────────
CUDA_DEVICE=0   # ← 修改为你要用的 GPU id（nvidia-smi 查看）

# ────────────────────────────────────────────────────────────
# ★ 序列选择
#   ALL_SEQS="true"  → 跑 data_root 下全部序列
#   ALL_SEQS="false" → 只跑 SEQUENCES 里指定的序列
# ────────────────────────────────────────────────────────────
ALL_SEQS="true"
SEQUENCES="video8 video75 video79 video97 video98 video102"    # ALL_SEQS=false 时生效

# ────────────────────────────────────────────────────────────
# ★ 运行模式开关
# ────────────────────────────────────────────────────────────
RUN_BASELINE="true"        # 是否跑无漂移纠偏基线
RUN_DRIFT_CORR="true"      # 是否跑带漂移纠偏版本
RUN_EVAL="true"            # 是否在两者完成后跑评测对比
SAVE_VIS="true"            # 是否保存可视化结果（关掉可节省磁盘空间）

# ────────────────────────────────────────────────────────────
# ★ 输出目录
# ────────────────────────────────────────────────────────────
OUTPUT_DIR="outputs"
EVAL_JSON="${OUTPUT_DIR}/eval_results.json"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ────────────────────────────────────────────────────────────
# 颜色定义
# ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────
log_info()  { echo -e "${CYAN}[INFO]  $*${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]    $*${NC}"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $*${NC}"; }
log_err()   { echo -e "${RED}[ERR]   $*${NC}"; }
log_head()  { echo -e "\n${YELLOW}${BOLD}$*${NC}"; }
log_sep()   { echo -e "${YELLOW}──────────────────────────────────────────────────${NC}"; }

format_duration() {
    local secs=$1
    printf "%dh %02dm %02ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

# ────────────────────────────────────────────────────────────
# 显示配置摘要
# ────────────────────────────────────────────────────────────
log_head "══════════════ CoralSAM-Track 配置摘要 ══════════════"
log_info "项目根目录 : ${PROJECT_ROOT}"
log_info "Python     : ${PYTHON}"
log_info "Config     : ${CONFIG}"
log_info "CUDA 设备  : GPU ${CUDA_DEVICE}"
log_info "序列模式   : $([ "$ALL_SEQS" = "true" ] && echo "全部序列" || echo "${SEQUENCES}")"
log_info "基线推理   : ${RUN_BASELINE}"
log_info "漂移纠偏   : ${RUN_DRIFT_CORR}"
log_info "评测对比   : ${RUN_EVAL}"
log_info "保存可视化 : ${SAVE_VIS}"
log_info "输出目录   : ${OUTPUT_DIR}"
log_sep

# 切换到项目根目录
cd "${PROJECT_ROOT}" || { log_err "无法进入项目目录: ${PROJECT_ROOT}"; exit 1; }

# ────────────────────────────────────────────────────────────
# Step 0：环境检查
# ────────────────────────────────────────────────────────────
log_head "Step 0 / 环境检查"

# 检查 Python
if [ ! -f "${PYTHON}" ]; then
    log_warn "指定 Python 不存在: ${PYTHON}"
    log_warn "尝试使用系统 python3..."
    PYTHON=$(which python3)
    if [ -z "${PYTHON}" ]; then
        log_err "找不到 python3，请修改 PYTHON 路径后重试"
        exit 1
    fi
fi
PYTHON_VER=$(${PYTHON} --version 2>&1)
log_ok "Python: ${PYTHON_VER}"

# 检查 PyTorch + CUDA
${PYTHON} - <<'PYCHECK'
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA可用 : {cuda_ok}")
    print(f"  GPU      : {device_name}")
    if not cuda_ok:
        print("  WARNING  : CUDA不可用，将使用CPU（速度会很慢）")
except ImportError:
    print("  ERROR: PyTorch 未安装，请先运行 pip install torch torchvision")
    sys.exit(1)
PYCHECK
PYCHECK_EXIT=$?
[ $PYCHECK_EXIT -ne 0 ] && { log_err "PyTorch 检查失败，请先安装依赖"; exit 1; }

# 检查 SAM2
${PYTHON} -c "import sam2; print(f'  SAM2     : {sam2.__version__}')" 2>/dev/null \
    || log_warn "SAM2 未安装或版本较老（仍可继续，运行时会报错）"

# 检查数据集
DATA_ROOT=$(${PYTHON} -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('data_root', 'partial_coralvos/partial'))
" 2>/dev/null)
DATA_ROOT="${DATA_ROOT:-partial_coralvos/partial}"

if [ ! -d "${DATA_ROOT}/images" ]; then
    log_err "数据集目录不存在: ${DATA_ROOT}/images"
    log_err "请确认 configs/default.yaml 中 data_root 指向正确路径"
    exit 1
fi

SEQ_COUNT=$(ls -1 "${DATA_ROOT}/images" | wc -l)
log_ok "数据集: ${DATA_ROOT}  (${SEQ_COUNT} 个序列: $(ls ${DATA_ROOT}/images | tr '\n' ' '))"

# 检查 SAM2 权重
SAM2_CKPT=$(${PYTHON} -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('sam2', {}).get('checkpoint', 'checkpoints/sam2.1_hiera_large.pt'))
" 2>/dev/null)
SAM2_CKPT="${SAM2_CKPT:-checkpoints/sam2.1_hiera_large.pt}"

if [ ! -f "${SAM2_CKPT}" ]; then
    log_warn "SAM2 权重不存在: ${SAM2_CKPT}"
    log_warn "尝试自动下载 sam2.1_hiera_large.pt ..."
    mkdir -p checkpoints
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "${SAM2_CKPT}" \
        && log_ok "SAM2 权重下载成功" \
        || { log_err "下载失败，请手动下载后放到 checkpoints/ 目录"; exit 1; }
fi
log_ok "SAM2 权重: ${SAM2_CKPT}"
log_sep

# ────────────────────────────────────────────────────────────
# 构建公共参数
# ────────────────────────────────────────────────────────────
SEQ_ARGS=""
if [ "$ALL_SEQS" = "false" ]; then
    SEQ_ARGS=""
    for SEQ in $SEQUENCES; do
        SEQ_ARGS="$SEQ_ARGS --seq $SEQ"
    done
    # demo.py 只接受单个 --seq，需要用 --all 或逐序列循环
    # 此处改为构建列表，在下方循环处理
    :
fi

# ────────────────────────────────────────────────────────────
# 内部函数：运行 demo.py（单次执行）
# ────────────────────────────────────────────────────────────
run_demo() {
    local mode_label=$1   # "基线" 或 "漂移纠偏"
    local drift_flag=$2   # "" 或 "--no_drift_correction"
    local log_file=$3

    local t_start=$(date +%s)
    log_head "推理 [$mode_label]  $(date '+%H:%M:%S')"

    if [ "$ALL_SEQS" = "true" ]; then
        # 一次性跑全部序列
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
        ${PYTHON} scripts/demo.py \
            --all \
            --config "${CONFIG}" \
            ${drift_flag} \
            --output "${OUTPUT_DIR}" \
            2>&1 | tee "${log_file}"
        local exit_code=${PIPESTATUS[0]}
    else
        # 逐序列循环（支持指定序列列表）
        local exit_code=0
        for SEQ in $SEQUENCES; do
            log_info "  处理序列: ${SEQ}"
            CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
            ${PYTHON} scripts/demo.py \
                --seq "${SEQ}" \
                --config "${CONFIG}" \
                ${drift_flag} \
                --output "${OUTPUT_DIR}" \
                2>&1 | tee -a "${log_file}"
            local ec=${PIPESTATUS[0]}
            [ $ec -ne 0 ] && exit_code=$ec
        done
    fi

    local t_end=$(date +%s)
    local dur=$((t_end - t_start))

    if [ $exit_code -eq 0 ]; then
        log_ok "[$mode_label] 推理完成  耗时: $(format_duration $dur)"
    else
        log_err "[$mode_label] 推理失败 (exit=$exit_code)  耗时: $(format_duration $dur)"
    fi
    return $exit_code
}

# ────────────────────────────────────────────────────────────
# Step 1：基线推理（无漂移纠偏）
# ────────────────────────────────────────────────────────────
FAILED_STEPS=""
TOTAL_START=$(date +%s)

if [ "$RUN_BASELINE" = "true" ]; then
    run_demo "基线（无漂移纠偏）" "--no_drift_correction" "${LOG_DIR}/baseline.log"
    if [ $? -eq 0 ]; then
        log_ok "基线 masks 已保存 → ${OUTPUT_DIR}/baseline/masks/"
    else
        FAILED_STEPS="$FAILED_STEPS baseline"
        log_err "基线推理失败，查看日志: ${LOG_DIR}/baseline.log"
    fi
    log_sep
fi

# ────────────────────────────────────────────────────────────
# Step 2：带漂移纠偏推理
# ────────────────────────────────────────────────────────────
if [ "$RUN_DRIFT_CORR" = "true" ]; then
    run_demo "漂移纠偏" "" "${LOG_DIR}/drift_corr.log"
    if [ $? -eq 0 ]; then
        log_ok "漂移纠偏 masks 已保存 → ${OUTPUT_DIR}/with_drift_corr/masks/"
    else
        FAILED_STEPS="$FAILED_STEPS drift_corr"
        log_err "漂移纠偏推理失败，查看日志: ${LOG_DIR}/drift_corr.log"
    fi
    log_sep
fi

# ────────────────────────────────────────────────────────────
# Step 3：评测对比（需要两者均已完成）
# ────────────────────────────────────────────────────────────
if [ "$RUN_EVAL" = "true" ]; then
    log_head "Step 3 / 评测对比  $(date '+%H:%M:%S')"
    t_eval_start=$(date +%s)

    PRED_DIR="${OUTPUT_DIR}/with_drift_corr/masks"
    BASE_DIR="${OUTPUT_DIR}/baseline/masks"

    # 检查预测目录是否存在
    EVAL_ARGS="--pred_dir ${PRED_DIR} --config ${CONFIG} --output_json ${EVAL_JSON}"
    if [ -d "${BASE_DIR}" ]; then
        EVAL_ARGS="$EVAL_ARGS --baseline_dir ${BASE_DIR}"
        log_info "对比模式：with_drift_corr vs baseline"
    else
        log_warn "baseline 目录不存在，仅评测 with_drift_corr"
    fi

    if [ ! -d "${PRED_DIR}" ]; then
        log_warn "预测目录不存在: ${PRED_DIR}，跳过评测"
    else
        ${PYTHON} scripts/eval.py ${EVAL_ARGS} \
            2>&1 | tee "${LOG_DIR}/eval.log"
        eval_exit=${PIPESTATUS[0]}

        t_eval_end=$(date +%s)
        eval_dur=$((t_eval_end - t_eval_start))

        if [ $eval_exit -eq 0 ]; then
            log_ok "评测完成  耗时: $(format_duration $eval_dur)"
            log_ok "结果 JSON: ${EVAL_JSON}"
        else
            FAILED_STEPS="$FAILED_STEPS eval"
            log_err "评测失败 (exit=$eval_exit)"
        fi
    fi
    log_sep
fi

# ────────────────────────────────────────────────────────────
# 总结
# ────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_DUR=$((TOTAL_END - TOTAL_START))

log_head "══════════════════ 运行总结 ══════════════════"
log_info "总耗时    : $(format_duration $TOTAL_DUR)"
log_info "输出目录  : ${OUTPUT_DIR}/"
log_info "  基线 masks : ${OUTPUT_DIR}/baseline/masks/"
log_info "  纠偏 masks : ${OUTPUT_DIR}/with_drift_corr/masks/"
log_info "  可视化    : ${OUTPUT_DIR}/*/vis/"
log_info "  评测 JSON  : ${EVAL_JSON}"
log_info "日志目录  : ${LOG_DIR}/"

if [ -n "$FAILED_STEPS" ]; then
    log_err "以下步骤失败：$FAILED_STEPS"
    log_err "查看对应日志: ${LOG_DIR}/"
    exit 1
else
    log_ok "全部步骤完成！"
fi
log_sep
