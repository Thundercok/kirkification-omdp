#!/usr/bin/env bash
# ==============================================================================
# Master Execution Pipeline for:
# "The Kirkification Effect: Quantifying Organic Memetic Data Pollution
#  and Morphological Feature Drift in Latent Diffusion Models"
# ==============================================================================

set -euo pipefail

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

CONFIG_FILE="config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log_step() {
    echo -e "\n${CYAN}================================================================${NC}"
    echo -e "${GREEN}[OMDP STEP $1]${NC} $2"
    echo -e "${CYAN}================================================================${NC}"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: ./run_pipeline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all             Run the complete end-to-end experimental pipeline (Steps 1-7)"
    echo "  --step <1-7>      Execute an individual pipeline step:"
    echo "                      1: Synthesize Kirkified Dataset (01_synthesize_kirk.py)"
    echo "                      2: Calculate Baseline Metrics (02_metrics.py)"
    echo "                      3: Mix Controlled Datasets (03_mix_datasets.py)"
    echo "                      4: Train LoRA Across Alpha Tiers (04_train_lora.py)"
    echo "                      5: Batch Inference & Evaluation (05_generate_eval.py)"
    echo "                      6: Apply OMDP Defense Sanitizer (06_sanitizer_filter.py)"
    echo "                      7: Generate IEEE-Ready Figures (07_plot_results.py)"
    echo "  --smoke-test      Run end-to-end verification with mock data (dry-run mode)"
    echo "  --mock-data       Generate synthetic portraits for local dry run"
    echo "  --help            Display this help message"
    echo ""
}

check_environment() {
    log_info "Verifying execution environment..."
    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        log_error "Python interpreter '$PYTHON_BIN' not found in PATH."
        exit 1
    fi

    # Create root directories
    mkdir -p data/clean data/poisoned data/reference_target data/mixed data/sanitized checkpoints eval_outputs figures results
}

# ------------------------------------------------------------------------------
# PIPELINE STEP DEFINITIONS
# ------------------------------------------------------------------------------

step_01_synthesize() {
    log_step "1" "Synthesizing Poisoned Kirkified Facial Portraits"
    "$PYTHON_BIN" src/01_synthesize_kirk.py --config "$CONFIG_FILE"
}

step_02_metrics() {
    log_step "2" "Computing Quantitative Facial Proportions & Identity Embeddings"
    log_info "Evaluating clean reference cohort..."
    "$PYTHON_BIN" src/02_metrics.py --config "$CONFIG_FILE" --eval_dir data/clean --out_json results/clean_metrics.json

    if [ -d "data/poisoned" ] && [ "$(ls -A data/poisoned 2>/dev/null)" ]; then
        log_info "Evaluating poisoned cohort..."
        "$PYTHON_BIN" src/02_metrics.py --config "$CONFIG_FILE" --eval_dir data/poisoned --out_json results/poisoned_metrics.json
    fi
}

step_03_mix() {
    log_step "3" "Constructing Controlled Contamination Mixtures (alpha in [0.0, 0.25])"
    "$PYTHON_BIN" src/03_mix_datasets.py --config "$CONFIG_FILE"
}

step_04_train() {
    log_step "4" "Fine-Tuning LoRA on Latent Diffusion Model across Alpha Tiers"
    for alpha_dir in data/mixed/alpha_*; do
        if [ -d "$alpha_dir" ]; then
            alpha_tag=$(basename "$alpha_dir")
            ckpt_out="checkpoints/lora_${alpha_tag}"
            log_info "Training LoRA for condition: ${alpha_tag} -> ${ckpt_out}"
            "$PYTHON_BIN" src/04_train_lora.py --config "$CONFIG_FILE" --data_dir "$alpha_dir" --output_dir "$ckpt_out"
        fi
    done
}

step_05_generate_eval() {
    log_step "5" "Executing Batch Inference and Morphological Evaluation"
    "$PYTHON_BIN" src/05_generate_eval.py --config "$CONFIG_FILE"
}

step_06_defense() {
    log_step "6" "Executing Data-Centric OMDP Outlier Defense Sanitizer"
    # Run defense against the alpha_0.10 contaminated split as reference benchmark
    TARGET_CONTAMINATED="data/mixed/alpha_0.10"
    if [ ! -d "$TARGET_CONTAMINATED" ]; then
        TARGET_CONTAMINATED="data/poisoned"
    fi
    "$PYTHON_BIN" src/06_sanitizer_filter.py \
        --config "$CONFIG_FILE" \
        --input_dir "$TARGET_CONTAMINATED" \
        --sanitized_dir "data/sanitized" \
        --out_json "results/sanitizer_audit.json"
}

step_07_plot() {
    log_step "7" "Rendering Camera-Ready Figures & Visualizations"
    "$PYTHON_BIN" src/07_plot_results.py --config "$CONFIG_FILE"
}

run_smoke_test() {
    log_step "SMOKE-TEST" "Executing Rapid End-to-End Pipeline Verification"
    log_info "Generating mock synthetic dataset..."
    "$PYTHON_BIN" src/utils/generate_mock_data.py --num_clean 8 --num_ref 2

    log_info "Running step 1 (Kirkification synthesis on mock data)..."
    "$PYTHON_BIN" src/01_synthesize_kirk.py --limit 4

    log_info "Running step 2 (Metrics verification)..."
    "$PYTHON_BIN" src/02_metrics.py --eval_dir data/clean --out_json results/smoke_metrics.json

    log_info "Running step 3 (Dataset mixing with total_samples=6)..."
    "$PYTHON_BIN" src/03_mix_datasets.py --total_samples 6

    log_info "Running step 5 (Evaluation dry-run)..."
    "$PYTHON_BIN" src/05_generate_eval.py --dry_run

    log_info "Running step 6 (Sanitizer defense filter)..."
    "$PYTHON_BIN" src/06_sanitizer_filter.py --input_dir data/poisoned --sanitized_dir data/sanitized

    log_info "Running step 7 (Figures & qualitative grid)..."
    "$PYTHON_BIN" src/07_plot_results.py

    echo -e "\n${GREEN}[SUCCESS] Smoke test completed successfully! All scripts verified.${NC}\n"
}

# ------------------------------------------------------------------------------
# CLI ENTRYPOINT
# ------------------------------------------------------------------------------

check_environment

if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            step_01_synthesize
            step_02_metrics
            step_03_mix
            step_04_train
            step_05_generate_eval
            step_06_defense
            step_07_plot
            echo -e "\n${GREEN}[ALL STEPS COMPLETED SUCCESSFULLY]${NC}\n"
            exit 0
            ;;
        --step)
            shift
            case "${1:-}" in
                1) step_01_synthesize ;;
                2) step_02_metrics ;;
                3) step_03_mix ;;
                4) step_04_train ;;
                5) step_05_generate_eval ;;
                6) step_06_defense ;;
                7) step_07_plot ;;
                *) log_error "Invalid step number: ${1:-}. Choose 1-7."; exit 1 ;;
            esac
            exit 0
            ;;
        --smoke-test)
            run_smoke_test
            exit 0
            ;;
        --mock-data)
            "$PYTHON_BIN" src/utils/generate_mock_data.py
            exit 0
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
    shift
done
