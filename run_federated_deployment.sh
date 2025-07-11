#!/bin/bash

##############################################################################
# Enhanced Federated nnUNet Implementation using Flower SuperNode/SuperLink
# Supports modality-aware federated averaging, dataset selection, model saving, and validation
#
# Bash implementation of run_federated_deployment.py for native terminal process management
##############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSES=()
CLEANUP_DONE=false

# Default values
DATASET="Dataset005_Prostate"
CLIENTS=2
ROUNDS=3
LOCAL_EPOCHS=1
MODE="run"
SUPERLINK_HOST="127.0.0.1"
SUPERLINK_PORT=9091
NODE_ID=0
PARTITION_ID=0
INSECURE=true
ENABLE_MODALITY_AGGREGATION=false
MODALITY_WEIGHTS=""
VALIDATE=true
NO_VALIDATE=false
VALIDATION_FREQUENCY=1
OUTPUT_DIR="federated_models"
SAVE_FREQUENCY=1
GPU=0
CLIENT_DATASETS=""
LIST_DATASETS=false
VALIDATE_DATASETS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

##############################################################################
# Utility Functions
##############################################################################

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

show_usage() {
    cat << EOF
Enhanced Federated nnUNet Implementation using Flower SuperNode/SuperLink Deployment

USAGE:
    $0 [OPTIONS]

OPTIONS:
    Dataset selection:
        --dataset DATASET               Dataset name (default: Dataset005_Prostate)
        --list-datasets                 List available datasets and exit
        --client-datasets JSON          JSON mapping of client IDs to datasets
        --validate-datasets             Validate dataset compatibility for multi-dataset federation

    Training configuration:
        --clients NUM                   Number of federated clients (default: 2)
        --rounds NUM                    Number of federated learning rounds (default: 3)
        --local-epochs NUM              Local epochs per client per round (default: 1)

    Deployment configuration:
        --mode MODE                     Deployment mode: superlink, supernode, or run (default: run)
        --superlink-host HOST           SuperLink host address (default: 127.0.0.1)
        --superlink-port PORT           SuperLink port (default: 9091)
        --node-id ID                    SuperNode ID for client mode (default: 0)
        --partition-id ID               Client partition ID (default: 0)
        --insecure                      Use insecure connection (default: true)

    Modality-aware aggregation:
        --enable-modality-aggregation   Enable modality-aware federated averaging
        --modality-weights JSON         JSON string of modality weights

    Validation options:
        --validate                      Run validation during training (default: true)
        --no-validate                   Skip validation (faster training)
        --validation-frequency NUM      Validate every N rounds (default: 1)

    Model saving options:
        --output-dir DIR                Output directory for saved models (default: federated_models)
        --save-frequency NUM            Save models every N rounds (default: 1)

    System configuration:
        --gpu ID                        GPU device ID to use (default: 0)

    Help:
        --help, -h                      Show this help message

EXAMPLES:
    # Basic deployment with 2 clients
    $0 --dataset Dataset005_Prostate --clients 2 --rounds 3

    # With modality-aware aggregation
    $0 --dataset Dataset005_Prostate --clients 2 --rounds 3 \\
        --enable-modality-aggregation \\
        --modality-weights '{"CT": 0.6, "MR": 0.4}'

    # Multi-dataset federation
    $0 --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}' \\
        --clients 2 --rounds 5 --enable-modality-aggregation

    # Manual deployment modes
    $0 --mode superlink                    # Start SuperLink only
    $0 --mode supernode --node-id 0        # Start SuperNode only
    $0 --mode run                          # Full automated deployment

EOF
}

##############################################################################
# Argument Parsing
##############################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # Dataset selection
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --list-datasets)
                LIST_DATASETS=true
                shift
                ;;
            --client-datasets)
                CLIENT_DATASETS="$2"
                shift 2
                ;;
            --validate-datasets)
                VALIDATE_DATASETS=true
                shift
                ;;
            
            # Training configuration
            --clients)
                CLIENTS="$2"
                shift 2
                ;;
            --rounds)
                ROUNDS="$2"
                shift 2
                ;;
            --local-epochs)
                LOCAL_EPOCHS="$2"
                shift 2
                ;;
            
            # Deployment configuration
            --mode)
                MODE="$2"
                if [[ ! "$MODE" =~ ^(superlink|supernode|run)$ ]]; then
                    log_error "Invalid mode: $MODE. Must be one of: superlink, supernode, run"
                    exit 1
                fi
                shift 2
                ;;
            --superlink-host)
                SUPERLINK_HOST="$2"
                shift 2
                ;;
            --superlink-port)
                SUPERLINK_PORT="$2"
                shift 2
                ;;
            --node-id)
                NODE_ID="$2"
                shift 2
                ;;
            --partition-id)
                PARTITION_ID="$2"
                shift 2
                ;;
            --insecure)
                INSECURE=true
                shift
                ;;
            
            # Modality-aware aggregation
            --enable-modality-aggregation)
                ENABLE_MODALITY_AGGREGATION=true
                shift
                ;;
            --modality-weights)
                MODALITY_WEIGHTS="$2"
                shift 2
                ;;
            
            # Validation options
            --validate)
                VALIDATE=true
                shift
                ;;
            --no-validate)
                NO_VALIDATE=true
                shift
                ;;
            --validation-frequency)
                VALIDATION_FREQUENCY="$2"
                shift 2
                ;;
            
            # Model saving options
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --save-frequency)
                SAVE_FREQUENCY="$2"
                shift 2
                ;;
            
            # System configuration
            --gpu)
                GPU="$2"
                shift 2
                ;;
            
            # Help
            --help|-h)
                show_usage
                exit 0
                ;;
            
            # Unknown option
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

##############################################################################
# Dataset Management Functions
##############################################################################

list_available_datasets() {
    local preproc_root="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
    
    if [[ ! -d "$preproc_root" ]]; then
        log_error "nnUNet preprocessed directory not found: $preproc_root"
        log_info "Set nnUNet_preprocessed environment variable or ensure the directory exists"
        return 1
    fi
    
    log_info "Available datasets in $preproc_root:"
    local count=0
    
    for item in "$preproc_root"/Dataset*; do
        if [[ -d "$item" ]]; then
            local dataset_name=$(basename "$item")
            local plans_file="$item/nnUNetPlans.json"
            local dataset_file="$item/dataset.json"
            
            if [[ -f "$plans_file" && -f "$dataset_file" ]]; then
                count=$((count + 1))
                echo "  $count. $dataset_name"
            fi
        fi
    done
    
    if [[ $count -eq 0 ]]; then
        log_warning "No valid datasets found in $preproc_root"
    fi
}

validate_dataset() {
    local dataset_name="$1"
    local preproc_root="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
    local dataset_path="$preproc_root/$dataset_name"
    
    if [[ ! -d "$dataset_path" ]]; then
        log_error "Dataset directory not found: $dataset_path"
        return 1
    fi
    
    local required_files=("nnUNetPlans.json" "dataset.json")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$dataset_path/$file" ]]; then
            log_error "Required file missing: $dataset_path/$file"
            return 1
        fi
    done
    
    log_success "Dataset '$dataset_name' is valid"
    return 0
}

parse_client_datasets() {
    local json_string="$1"
    
    # Basic JSON parsing - for more complex validation, we might need to call Python
    if ! echo "$json_string" | python3 -m json.tool > /dev/null 2>&1; then
        log_error "Invalid JSON format for client-datasets"
        return 1
    fi
    
    log_success "Client-datasets JSON is valid"
    return 0
}

##############################################################################
# Setup Functions
##############################################################################

setup_environment() {
    log_info "Setting up environment variables..."
    
    # Core environment variables
    export CUDA_VISIBLE_DEVICES="$GPU"
    export TORCH_COMPILE_DISABLE="1"
    export OMP_NUM_THREADS="1"
    export MKL_NUM_THREADS="1" 
    export NUMEXPR_NUM_THREADS="1"
    export nnUNet_n_proc_DA="1"
    export TOKENIZERS_PARALLELISM="false"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    export TORCHINDUCTOR_COMPILE_THREADS="1"
    
    log_success "Environment variables configured"
}

setup_federation_environment() {
    log_info "Setting up federated learning environment variables..."
    
    # Basic federation settings
    export TASK_NAME="$DATASET"
    export NUM_CLIENTS="$CLIENTS"
    export NUM_TRAINING_ROUNDS="$ROUNDS"
    export LOCAL_EPOCHS="$LOCAL_EPOCHS"
    export OUTPUT_DIR="$OUTPUT_DIR"
    export SAVE_FREQUENCY="$SAVE_FREQUENCY"
    
    # Validation settings
    if [[ "$NO_VALIDATE" == "true" ]]; then
        export VALIDATE_ENABLED="false"
    else
        export VALIDATE_ENABLED="$VALIDATE"
    fi
    export VALIDATION_FREQUENCY="$VALIDATION_FREQUENCY"
    
    # Modality-aware aggregation settings
    if [[ "$ENABLE_MODALITY_AGGREGATION" == "true" ]]; then
        export ENABLE_MODALITY_AGGREGATION="true"
        if [[ -n "$MODALITY_WEIGHTS" ]]; then
            export MODALITY_WEIGHTS="$MODALITY_WEIGHTS"
        fi
    fi
    
    # Multi-dataset settings
    if [[ -n "$CLIENT_DATASETS" ]]; then
        export CLIENT_DATASETS="$CLIENT_DATASETS"
        log_info "Multi-dataset federation enabled"
        
        # Parse and set individual client environment variables
        # This is a simplified version - for full compatibility, might need Python helper
        local client_count=0
        while read -r line; do
            if [[ "$line" =~ \"([0-9]+)\".*\"([^\"]+)\" ]]; then
                local client_id="${BASH_REMATCH[1]}"
                local dataset_name="${BASH_REMATCH[2]}"
                export "CLIENT_${client_id}_DATASET=$dataset_name"
                client_count=$((client_count + 1))
            fi
        done <<< "$CLIENT_DATASETS"
        
        log_info "Configured $client_count client-dataset mappings"
    fi
    
    log_success "Federation environment configured"
}

##############################################################################
# Process Management Functions
##############################################################################

start_superlink() {
    log_info "Starting SuperLink server..."
    
    local cmd=(
        "flower-superlink"
        "--insecure"
    )
    
    log_info "Command: ${cmd[*]}"
    
    # Start SuperLink in background and capture PID
    "${cmd[@]}" &
    local pid=$!
    PROCESSES+=($pid)
    
    log_success "SuperLink started with PID: $pid"
    return $pid
}

start_supernode() {
    local node_id=${1:-$NODE_ID}
    local partition_id=${2:-$((PARTITION_ID + node_id))}
    
    log_info "Starting SuperNode $node_id (partition: $partition_id)..."
    
    # Use different ClientApp API addresses for each SuperNode
    local clientapp_port=$((9094 + node_id))
    
    local cmd=(
        "flower-supernode"
        "--insecure"
        "--superlink" "${SUPERLINK_HOST}:9092"
        "--clientappio-api-address" "${SUPERLINK_HOST}:${clientapp_port}"
        "--node-config" "partition-id=${partition_id}"
    )
    
    log_info "Command: ${cmd[*]}"
    
    # Set client-specific environment variables
    env CLIENT_ID="$node_id" PARTITION_ID="$partition_id" "${cmd[@]}" &
    local pid=$!
    PROCESSES+=($pid)
    
    log_success "SuperNode $node_id started with PID: $pid"
    return $pid
}

run_federation() {
    log_info "Running federated learning..."
    
    local cmd=(
        "flwr" "run" "." "deployment"
    )
    
    log_info "Command: ${cmd[*]}"
    
    # Run federation command
    "${cmd[@]}"
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Federated learning completed successfully"
    else
        log_error "Federated learning failed with exit code: $exit_code"
    fi
    
    return $exit_code
}

##############################################################################
# Signal Handling and Cleanup
##############################################################################

cleanup_processes() {
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi
    
    CLEANUP_DONE=true
    log_info "Cleaning up processes..."
    
    for pid in "${PROCESSES[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Terminating process $pid..."
            if ! kill "$pid" 2>/dev/null; then
                log_warning "Failed to terminate process $pid"
            else
                # Wait for process to terminate gracefully
                local count=0
                while kill -0 "$pid" 2>/dev/null && [[ $count -lt 5 ]]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    log_warning "Force killing process $pid..."
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        fi
    done
    
    log_success "Cleanup completed"
}

signal_handler() {
    local signal="$1"
    log_warning "Received signal $signal"
    cleanup_processes
    exit 0
}

# Set up signal handlers
trap 'signal_handler SIGINT' SIGINT
trap 'signal_handler SIGTERM' SIGTERM

##############################################################################
# Main Script
##############################################################################

main() {
    log_header "=== Flower SuperNode/SuperLink based nnUNet Federated Learning ==="
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Set up basic environment
    setup_environment
    
    # Handle list datasets option
    if [[ "$LIST_DATASETS" == "true" ]]; then
        list_available_datasets
        exit 0
    fi
    
    # Handle multi-dataset configuration
    if [[ -n "$CLIENT_DATASETS" ]]; then
        log_info "Multi-dataset federation mode"
        
        # Validate JSON format
        if ! parse_client_datasets "$CLIENT_DATASETS"; then
            log_error "Invalid client-datasets configuration"
            exit 1
        fi
        
        # For complex multi-dataset validation, we'd call Python helper
        if [[ "$VALIDATE_DATASETS" == "true" ]]; then
            log_info "Running dataset compatibility validation..."
            if command -v python3 >/dev/null 2>&1 && [[ -f "dataset_compatibility.py" ]]; then
                if ! python3 -c "
import json
from dataset_compatibility import create_multi_dataset_config
client_datasets = json.loads('$CLIENT_DATASETS')
preproc_root = '${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}'
try:
    config = create_multi_dataset_config(client_datasets, preproc_root)
    print('✅ Multi-dataset validation passed')
except Exception as e:
    print(f'❌ Multi-dataset validation failed: {e}')
    exit(1)
                "; then
                    log_error "Multi-dataset validation failed"
                    exit 1
                fi
            else
                log_warning "Dataset compatibility validation not available"
            fi
        fi
        
        # Automatically enable modality aggregation for multi-dataset
        if [[ "$ENABLE_MODALITY_AGGREGATION" != "true" ]]; then
            log_info "Automatically enabling modality-aware aggregation for multi-dataset federation"
            ENABLE_MODALITY_AGGREGATION=true
        fi
        
    else
        # Single dataset mode
        log_info "Single-dataset federation mode: $DATASET"
        
        # Validate single dataset
        if ! validate_dataset "$DATASET"; then
            log_error "Dataset validation failed"
            list_available_datasets
            exit 1
        fi
    fi
    
    # Set up federation environment
    setup_federation_environment
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Display configuration
    log_info "Configuration:"
    if [[ -n "$CLIENT_DATASETS" ]]; then
        log_info "  Federation Type: Multi-Dataset"
        log_info "  Client-Dataset Mapping: $CLIENT_DATASETS"
    else
        log_info "  Federation Type: Single-Dataset ($DATASET)"
    fi
    log_info "  Clients: $CLIENTS"
    log_info "  Rounds: $ROUNDS"
    log_info "  Local epochs: $LOCAL_EPOCHS"
    log_info "  SuperLink: ${SUPERLINK_HOST}:${SUPERLINK_PORT}"
    log_info "  Modality-aware aggregation: $([ "$ENABLE_MODALITY_AGGREGATION" == "true" ] && echo "enabled" || echo "disabled")"
    log_info "  Validation: $([ "$NO_VALIDATE" == "true" ] && echo "disabled" || echo "enabled")"
    log_info "  Output directory: $OUTPUT_DIR"
    
    # Execute based on mode
    case "$MODE" in
        "superlink")
            log_info "Mode: SuperLink only"
            start_superlink
            local superlink_pid=$?
            log_info "SuperLink running. Press Ctrl+C to stop."
            wait $superlink_pid
            ;;
            
        "supernode")
            log_info "Mode: SuperNode only (Node ID: $NODE_ID)"
            log_info "Waiting 2 seconds for SuperLink to be ready..."
            sleep 2
            
            start_supernode "$NODE_ID" "$PARTITION_ID"
            local supernode_pid=$?
            log_info "SuperNode running. Press Ctrl+C to stop."
            wait $supernode_pid
            ;;
            
        "run")
            log_info "Mode: Full automated deployment"
            
            # Start SuperLink
            start_superlink
            local superlink_pid=$?
            
            # Wait for SuperLink to be ready
            log_info "Waiting 3 seconds for SuperLink to be ready..."
            sleep 3
            
            # Start SuperNodes
            log_info "Starting $CLIENTS SuperNodes..."
            local supernode_pids=()
            for ((i=0; i<CLIENTS; i++)); do
                start_supernode "$i" "$((PARTITION_ID + i))"
                supernode_pids+=($?)
                sleep 1  # Stagger SuperNode startup
            done
            
            # Wait for SuperNodes to be ready
            log_info "Waiting 5 seconds for SuperNodes to be ready..."
            sleep 5
            
            # Run the federation
            log_success "Starting federated learning..."
            if run_federation; then
                log_success "Federated learning completed successfully!"
            else
                log_error "Federated learning failed"
                cleanup_processes
                exit 1
            fi
            ;;
            
        *)
            log_error "Invalid mode: $MODE"
            exit 1
            ;;
    esac
    
    # Cleanup on normal exit
    cleanup_processes
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi