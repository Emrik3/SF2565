#!/bin/bash

# Convenience script for feature_nn training
# Makes common tasks easier to run

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Feature NN - Quick Runner${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Print usage
print_usage() {
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  simple            - Train simple 2-layer network"
    echo "  default           - Train default 3-layer network"
    echo "  deep              - Train deep 5-layer network"
    echo "  sweep-lr          - Sweep learning rates"
    echo "  sweep-arch        - Sweep architectures"
    echo "  sweep-dropout     - Sweep dropout rates"
    echo "  compare           - Compare all experiments"
    echo "  help              - Show this help message"
    echo ""
}


# Train simple
run_simple() {
    echo -e "${YELLOW}Training simple network...${NC}"
    python src/train.py --mode simple
}

# Train default
run_default() {
    echo -e "${YELLOW}Training default network...${NC}"
    python src/train.py --mode default
}

# Train deep
run_deep() {
    echo -e "${YELLOW}Training deep network...${NC}"
    python src/train.py --mode deep
}

# Sweep learning rate
run_sweep_lr() {
    echo -e "${YELLOW}Sweeping learning rates...${NC}"
    python src/parameter_sweep.py --sweep learning_rate
}

# Sweep architecture
run_sweep_arch() {
    echo -e "${YELLOW}Sweeping architectures...${NC}"
    python src/parameter_sweep.py --sweep architecture
}

# Sweep dropout
run_sweep_dropout() {
    echo -e "${YELLOW}Sweeping dropout rates...${NC}"
    python src/parameter_sweep.py --sweep dropout
}

# Compare experiments
run_compare() {
    echo -e "${YELLOW}Comparing experiments...${NC}"
    python src/compare_experiments.py --plot --export comparison.csv
    echo -e "${GREEN}Results saved to comparison.csv${NC}"
    echo -e "${GREEN}Plots saved to experiment_comparison.png and experiment_bars.png${NC}"
}

# Main script
print_header

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    simple)
        run_simple
        ;;
    default)
        run_default
        ;;
    deep)
        run_deep
        ;;
    sweep-lr)
        run_sweep_lr
        ;;
    sweep-arch)
        run_sweep_arch
        ;;
    sweep-dropout)
        run_sweep_dropout
        ;;
    compare)
        run_compare
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        print_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
