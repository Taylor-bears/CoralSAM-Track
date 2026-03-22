#!/bin/bash
set -euo pipefail

# ============================================================
# CoralSAM-Track revision pipeline
# Runs the minimum revision package requested by reviewers:
#   1. Ablations: baseline / soft_only / no_gate / full
#   2. Optional external-method comparison from mask folders
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data1/baiyang/anaconda/envs/coralsam/bin/python"
CONFIG="configs/default.yaml"

# Use the conda env libstdc++ first on remote Linux servers
CONDA_LIB="/data1/baiyang/anaconda/envs/coralsam/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

# ------------------------------
# Runtime settings
# ------------------------------
CUDA_DEVICE=1
ALL_SEQS="true"
SEQUENCES="video8 video75 video79 video97 video98 video102"
MODES="baseline soft_only no_gate full"
SAVE_VIS="false"

# Optional external methods.
# Format: LABEL=/absolute/path/to/masks_root
# Example:
# EXTRA_METHODS=(
#   "CUTIE=/data1/xxx/cutie_outputs/masks"
#   "DEVA=/data1/xxx/deva_outputs/masks"
# )
EXTRA_METHODS=()

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="outputs/revision_${RUN_ID}"
REPORT_DIR="${OUTPUT_DIR}/revision_stats"
LOG_DIR="logs"
REVISION_LOG="${LOG_DIR}/revision_${RUN_ID}.log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]  $*${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]    $*${NC}"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $*${NC}"; }
log_err()   { echo -e "${RED}[ERR]   $*${NC}"; }

cd "${PROJECT_ROOT}" || { log_err "Cannot enter project root: ${PROJECT_ROOT}"; exit 1; }

log_info "Project root : ${PROJECT_ROOT}"
log_info "Python       : ${PYTHON}"
log_info "Config       : ${CONFIG}"
log_info "CUDA device  : ${CUDA_DEVICE}"
log_info "Run ID       : ${RUN_ID}"
log_info "Output dir   : ${OUTPUT_DIR}"
log_info "Modes        : ${MODES}"
if [ "${ALL_SEQS}" = "true" ]; then
    log_info "Sequences    : all sequences"
else
    log_info "Sequences    : ${SEQUENCES}"
fi
if [ ${#EXTRA_METHODS[@]} -gt 0 ]; then
    log_info "Extra method count: ${#EXTRA_METHODS[@]}"
else
    log_info "Extra methods: none"
fi

if [ ! -f "${PYTHON}" ]; then
    log_err "Python not found: ${PYTHON}"
    exit 1
fi

DATA_ROOT="$(${PYTHON} -c "
import yaml
with open('${CONFIG}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('data_root', 'partial_coralvos/partial'))
" 2>/dev/null)"
DATA_ROOT="${DATA_ROOT:-partial_coralvos/partial}"

if [ ! -d "${DATA_ROOT}/images" ]; then
    log_err "Dataset directory not found: ${DATA_ROOT}/images"
    exit 1
fi

CMD=(
    "${PYTHON}" "scripts/revision_experiments.py"
    "--config" "${CONFIG}"
    "--output" "${OUTPUT_DIR}"
    "--run_id" "${RUN_ID}"
    "--device" "cuda"
    "--modes"
)

for mode in ${MODES}; do
    CMD+=("${mode}")
done

if [ "${ALL_SEQS}" = "true" ]; then
    CMD+=("--all")
else
    CMD+=("--seq")
    for seq in ${SEQUENCES}; do
        CMD+=("${seq}")
    done
fi

if [ "${SAVE_VIS}" = "true" ]; then
    CMD+=("--save_vis")
fi

for item in "${EXTRA_METHODS[@]}"; do
    CMD+=("--extra_method" "${item}")
done

log_info "Starting revision pipeline..."
echo "$(date '+%Y-%m-%d %H:%M:%S') [START] revision pipeline" > "${REVISION_LOG}"
echo "Command: ${CMD[*]}" >> "${REVISION_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}" 2>&1 | tee -a "${REVISION_LOG}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_err "Revision pipeline failed (exit=${EXIT_CODE})"
    log_err "See log: ${REVISION_LOG}"
    exit ${EXIT_CODE}
fi

log_ok "Revision pipeline completed"
log_ok "Report dir : ${REPORT_DIR}"
log_ok "Main CSV   : ${REPORT_DIR}/ablation_overview.csv"
log_ok "Compare CSV: ${REPORT_DIR}/comparison_overview.csv"
log_ok "Markdown   : ${REPORT_DIR}/revision_report.md"
log_ok "Log file   : ${REVISION_LOG}"
