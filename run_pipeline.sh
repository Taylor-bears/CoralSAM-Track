#!/bin/bash
# ============================================================
# CoralSAM-Track 完整流水线脚本（Linux 远程服务器）
# 功能：环境检查 → 基线推理 → 漂移纠偏推理 → 评测对比
# ============================================================

# ────────────────────────────────────────────────────────────
# ★ 基本路径配置
# ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data1/baiyang/anaconda/envs/coralsam/bin/python"
CONFIG="configs/default.yaml"

# 确保 conda 环境的 libstdc++ 优先于系统版本（修复 GLIBCXX 版本冲突）
CONDA_LIB="/data1/baiyang/anaconda/envs/coralsam/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

# ────────────────────────────────────────────────────────────
# ★ 并行模式
#   "off"      → 串行执行（安全，适合只有一张空卡时）
#   "same_gpu" → 两任务同一张卡并行（显存充足时，如 3090 空余 >18GB）
#   "dual_gpu" → 两任务分别用两张不同的卡（推荐，速度最快）
# ────────────────────────────────────────────────────────────
PARALLEL_MODE="off"   # off | same_gpu | dual_gpu

# 串行 / same_gpu 模式时使用的单卡
CUDA_DEVICE=3

# dual_gpu 模式时分别指定两张卡
CUDA_DEVICE_BASELINE=3     # baseline（无漂移纠偏）用这张卡（空余 24GB）
CUDA_DEVICE_DRIFT=0        # drift_corr（漂移纠偏）用这张卡（空余 24GB）

# ────────────────────────────────────────────────────────────
# ★ 序列选择
#   ALL_SEQS="true"  → 跑 data_root 下全部序列
#   ALL_SEQS="false" → 只跑 SEQUENCES 里指定的序列
# ────────────────────────────────────────────────────────────
ALL_SEQS="true"
SEQUENCES="video8 video75 video79 video97 video98 video102"

# ────────────────────────────────────────────────────────────
# ★ 运行模式开关
# ────────────────────────────────────────────────────────────
RUN_BASELINE="true"
RUN_DRIFT_CORR="true"
RUN_EVAL="true"
SAVE_VIS="true"

# ────────────────────────────────────────────────────────────
# ★ 时间戳 Run ID
# ────────────────────────────────────────────────────────────
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# ────────────────────────────────────────────────────────────
# ★ 输出目录
# ────────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/${RUN_ID}"
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
# 配置摘要
# ────────────────────────────────────────────────────────────
log_head "══════════════ CoralSAM-Track 配置摘要 ══════════════"
log_info "项目根目录  : ${PROJECT_ROOT}"
log_info "Python      : ${PYTHON}"
log_info "Config      : ${CONFIG}"
log_info "并行模式    : ${PARALLEL_MODE}"
if [ "${PARALLEL_MODE}" = "dual_gpu" ]; then
    log_info "GPU baseline: ${CUDA_DEVICE_BASELINE}"
    log_info "GPU drift   : ${CUDA_DEVICE_DRIFT}"
else
    log_info "CUDA 设备   : GPU ${CUDA_DEVICE}"
fi
log_info "序列模式    : $([ "$ALL_SEQS" = "true" ] && echo "全部序列" || echo "${SEQUENCES}")"
log_info "基线推理    : ${RUN_BASELINE}"
log_info "漂移纠偏    : ${RUN_DRIFT_CORR}"
log_info "评测对比    : ${RUN_EVAL}"
log_info "保存可视化  : ${SAVE_VIS}"
log_info "Run ID      : ${RUN_ID}"
log_info "输出目录    : ${OUTPUT_DIR}"
log_sep

cd "${PROJECT_ROOT}" || { log_err "无法进入项目目录: ${PROJECT_ROOT}"; exit 1; }

# ────────────────────────────────────────────────────────────
# Step 0：环境检查
# ────────────────────────────────────────────────────────────
log_head "Step 0 / 环境检查"

if [ ! -f "${PYTHON}" ]; then
    log_warn "指定 Python 不存在: ${PYTHON}，尝试系统 python3..."
    PYTHON=$(which python3)
    [ -z "${PYTHON}" ] && { log_err "找不到 python3"; exit 1; }
fi
log_ok "Python: $(${PYTHON} --version 2>&1)"

${PYTHON} - <<'PYCHECK'
import sys, torch
cuda_ok = torch.cuda.is_available()
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA可用 : {cuda_ok}")
if cuda_ok:
    import os
    dev = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"  GPU[0]   : {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING  : CUDA不可用，将使用CPU（速度很慢）")
PYCHECK
[ $? -ne 0 ] && { log_err "PyTorch 检查失败"; exit 1; }

${PYTHON} -c "import sam2; print(f'  SAM2     : {sam2.__version__}')" 2>/dev/null \
    || log_warn "SAM2 未安装或版本较老（仍可继续）"

DATA_ROOT=$(${PYTHON} -c "
import yaml
with open('${CONFIG}') as f: cfg = yaml.safe_load(f)
print(cfg.get('data_root', 'partial_coralvos/partial'))
" 2>/dev/null)
DATA_ROOT="${DATA_ROOT:-partial_coralvos/partial}"

[ ! -d "${DATA_ROOT}/images" ] && {
    log_err "数据集目录不存在: ${DATA_ROOT}/images"; exit 1; }

SEQ_COUNT=$(ls -1 "${DATA_ROOT}/images" | wc -l)
log_ok "数据集: ${DATA_ROOT}  (${SEQ_COUNT} 个序列: $(ls ${DATA_ROOT}/images | tr '\n' ' '))"

SAM2_CKPT=$(${PYTHON} -c "
import yaml
with open('${CONFIG}') as f: cfg = yaml.safe_load(f)
print(cfg.get('sam2', {}).get('checkpoint', 'checkpoints/sam2.1_hiera_large.pt'))
" 2>/dev/null)
SAM2_CKPT="${SAM2_CKPT:-checkpoints/sam2.1_hiera_large.pt}"

if [ ! -f "${SAM2_CKPT}" ]; then
    log_warn "SAM2 权重不存在，尝试自动下载..."
    mkdir -p checkpoints
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "${SAM2_CKPT}" \
        && log_ok "SAM2 权重下载成功" \
        || { log_err "下载失败，请手动下载"; exit 1; }
fi
log_ok "SAM2 权重: ${SAM2_CKPT}"
log_sep

# ────────────────────────────────────────────────────────────
# 日志文件路径
# ────────────────────────────────────────────────────────────
BASELINE_LOG="${LOG_DIR}/baseline_${RUN_ID}.log"
DRIFT_LOG="${LOG_DIR}/drift_corr_${RUN_ID}.log"
EVAL_LOG="${LOG_DIR}/eval_${RUN_ID}.log"

# ────────────────────────────────────────────────────────────
# 内部函数：串行模式 — 运行 demo.py 并实时打印到终端
# ────────────────────────────────────────────────────────────
run_demo_serial() {
    local mode_label=$1
    local gpu_device=$2
    local drift_flag=$3
    local log_file=$4
    local t_start=$(date +%s)

    log_head "推理 [${mode_label}]  GPU=${gpu_device}  $(date '+%H:%M:%S')"

    if [ "$ALL_SEQS" = "true" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_device} \
        ${PYTHON} scripts/demo.py \
            --all --config "${CONFIG}" ${drift_flag} \
            --output "${OUTPUT_DIR}" --run_id "${RUN_ID}" \
            2>&1 | tee "${log_file}"
        local ec=${PIPESTATUS[0]}
    else
        local ec=0
        for SEQ in $SEQUENCES; do
            log_info "  处理序列: ${SEQ}"
            CUDA_VISIBLE_DEVICES=${gpu_device} \
            ${PYTHON} scripts/demo.py \
                --seq "${SEQ}" --config "${CONFIG}" ${drift_flag} \
                --output "${OUTPUT_DIR}" --run_id "${RUN_ID}" \
                2>&1 | tee -a "${log_file}"
            local seq_ec=${PIPESTATUS[0]}
            [ $seq_ec -ne 0 ] && ec=$seq_ec
        done
    fi

    local dur=$(( $(date +%s) - t_start ))
    if [ $ec -eq 0 ]; then
        log_ok "[${mode_label}] 完成  耗时: $(format_duration $dur)"
    else
        log_err "[${mode_label}] 失败 (exit=$ec)  耗时: $(format_duration $dur)"
    fi
    return $ec
}

# ────────────────────────────────────────────────────────────
# 内部函数：并行模式 — 后台运行，只写日志文件
# ────────────────────────────────────────────────────────────
run_demo_bg() {
    local mode_label=$1
    local gpu_device=$2
    local drift_flag=$3
    local log_file=$4
    local t_start=$(date +%s)

    echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${mode_label}  GPU=${gpu_device}" > "${log_file}"

    if [ "$ALL_SEQS" = "true" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_device} \
        ${PYTHON} scripts/demo.py \
            --all --config "${CONFIG}" ${drift_flag} \
            --output "${OUTPUT_DIR}" --run_id "${RUN_ID}" \
            >> "${log_file}" 2>&1
        local ec=$?
    else
        local ec=0
        for SEQ in $SEQUENCES; do
            CUDA_VISIBLE_DEVICES=${gpu_device} \
            ${PYTHON} scripts/demo.py \
                --seq "${SEQ}" --config "${CONFIG}" ${drift_flag} \
                --output "${OUTPUT_DIR}" --run_id "${RUN_ID}" \
                >> "${log_file}" 2>&1
            local seq_ec=$?
            [ $seq_ec -ne 0 ] && ec=$seq_ec
        done
    fi

    local dur=$(( $(date +%s) - t_start ))
    if [ $ec -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE]  ${mode_label}  耗时: $(format_duration $dur)" >> "${log_file}"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL]  ${mode_label}  exit=$ec  耗时: $(format_duration $dur)" >> "${log_file}"
    fi
    return $ec
}

# ────────────────────────────────────────────────────────────
# Steps 1+2：推理（串行 or 并行）
# ────────────────────────────────────────────────────────────
FAILED_STEPS=""
TOTAL_START=$(date +%s)

if [ "${PARALLEL_MODE}" = "off" ]; then
    # ── 串行：按顺序逐个执行 ──────────────────────────────
    if [ "$RUN_BASELINE" = "true" ]; then
        run_demo_serial "基线（无漂移纠偏）" "${CUDA_DEVICE}" "--no_drift_correction" "${BASELINE_LOG}"
        [ $? -eq 0 ] && log_ok "基线 masks → ${OUTPUT_DIR}/baseline/masks/" \
                      || FAILED_STEPS="$FAILED_STEPS baseline"
        log_sep
    fi

    if [ "$RUN_DRIFT_CORR" = "true" ]; then
        run_demo_serial "漂移纠偏" "${CUDA_DEVICE}" "" "${DRIFT_LOG}"
        [ $? -eq 0 ] && log_ok "漂移纠偏 masks → ${OUTPUT_DIR}/with_drift_corr/masks/" \
                      || FAILED_STEPS="$FAILED_STEPS drift_corr"
        log_sep
    fi

else
    # ── 并行：两任务同时启动，等待双方结束 ───────────────
    if [ "${PARALLEL_MODE}" = "dual_gpu" ]; then
        GPU_BASELINE=${CUDA_DEVICE_BASELINE}
        GPU_DRIFT=${CUDA_DEVICE_DRIFT}
    else
        # same_gpu：共享同一张卡
        GPU_BASELINE=${CUDA_DEVICE}
        GPU_DRIFT=${CUDA_DEVICE}
    fi

    PARA_START=$(date +%s)
    log_head "Steps 1+2 / 并行推理  模式=${PARALLEL_MODE}  $(date '+%H:%M:%S')"

    PID_BASELINE=""
    PID_DRIFT=""

    if [ "$RUN_BASELINE" = "true" ]; then
        run_demo_bg "baseline" "${GPU_BASELINE}" "--no_drift_correction" "${BASELINE_LOG}" &
        PID_BASELINE=$!
        log_ok "baseline   已后台启动  GPU=${GPU_BASELINE}  PID=${PID_BASELINE}"
        log_info "  实时日志: tail -f ${BASELINE_LOG}"
    fi

    if [ "$RUN_DRIFT_CORR" = "true" ]; then
        run_demo_bg "drift_corr" "${GPU_DRIFT}" "" "${DRIFT_LOG}" &
        PID_DRIFT=$!
        log_ok "drift_corr 已后台启动  GPU=${GPU_DRIFT}  PID=${PID_DRIFT}"
        log_info "  实时日志: tail -f ${DRIFT_LOG}"
    fi

    log_sep
    log_info "两任务并行运行中，等待完成..."
    log_info "可在另一终端运行以下命令实时查看进度："
    log_info "  tail -f ${BASELINE_LOG}"
    log_info "  tail -f ${DRIFT_LOG}"
    log_sep

    EC_BASELINE=0
    EC_DRIFT=0

    if [ -n "${PID_BASELINE}" ]; then
        wait ${PID_BASELINE}
        EC_BASELINE=$?
        if [ $EC_BASELINE -eq 0 ]; then
            log_ok "baseline   完成  (PID=${PID_BASELINE})"
        else
            log_err "baseline   失败 (exit=${EC_BASELINE})"
            FAILED_STEPS="$FAILED_STEPS baseline"
        fi
    fi

    if [ -n "${PID_DRIFT}" ]; then
        wait ${PID_DRIFT}
        EC_DRIFT=$?
        if [ $EC_DRIFT -eq 0 ]; then
            log_ok "drift_corr 完成  (PID=${PID_DRIFT})"
        else
            log_err "drift_corr 失败 (exit=${EC_DRIFT})"
            FAILED_STEPS="$FAILED_STEPS drift_corr"
        fi
    fi

    PARA_DUR=$(( $(date +%s) - PARA_START ))
    log_ok "并行推理结束  实际耗时: $(format_duration $PARA_DUR)"
    log_sep
fi

# ────────────────────────────────────────────────────────────
# Step 3：评测对比
# ────────────────────────────────────────────────────────────
if [ "$RUN_EVAL" = "true" ]; then
    log_head "Step 3 / 评测对比  $(date '+%H:%M:%S')"
    t_eval_start=$(date +%s)

    PRED_DIR="${OUTPUT_DIR}/with_drift_corr/masks"
    BASE_DIR="${OUTPUT_DIR}/baseline/masks"

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
        ${PYTHON} scripts/eval.py ${EVAL_ARGS} 2>&1 | tee "${EVAL_LOG}"
        eval_exit=${PIPESTATUS[0]}
        eval_dur=$(( $(date +%s) - t_eval_start ))

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
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))

log_head "══════════════════ 运行总结 ══════════════════"
log_info "Run ID      : ${RUN_ID}"
log_info "并行模式    : ${PARALLEL_MODE}"
log_info "总耗时      : $(format_duration $TOTAL_DUR)"
log_info "输出目录    : ${OUTPUT_DIR}/"
log_info "  基线 masks  : ${OUTPUT_DIR}/baseline/masks/"
log_info "  纠偏 masks  : ${OUTPUT_DIR}/with_drift_corr/masks/"
log_info "  可视化      : ${OUTPUT_DIR}/*/vis/"
log_info "  评测 JSON   : ${EVAL_JSON}"
log_info "日志目录    : ${LOG_DIR}/"
log_info "  baseline    : ${BASELINE_LOG}"
log_info "  drift_corr  : ${DRIFT_LOG}"
log_info "  eval        : ${EVAL_LOG}"

if [ -n "$FAILED_STEPS" ]; then
    log_err "以下步骤失败：$FAILED_STEPS"
    exit 1
else
    log_ok "全部步骤完成！"
fi
log_sep
