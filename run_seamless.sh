#!/bin/bash
#
# run_seamless.sh
# Run SEAMLESS robustness analysis on a benchmark network
#
# Usage:
#   ./run_seamless.sh BA 1           # Run on BA_N256/BA_r1.edges
#   ./run_seamless.sh WS 42          # Run on WS_N256/WS_r42.edges
#   ./run_seamless.sh ER 10 --n-attacks 50  # Custom number of replicates
#
# Options:
#   --n-attacks N     Number of stochastic replicates (default: 10)
#   --m-min M         Minimum m for SEAMLESS (default: 1)
#   --m-max M         Maximum m for SEAMLESS (default: 20)
#   --protocols P     Space-separated list of protocols (default: all)
#   --zero-indexed    Use if network uses 0-indexed nodes
#

set -e

# Default parameters
N_ATTACKS=100
M_MIN=1
M_MAX=20
PROTOCOLS="random degree adap_degree betweenness adap_betweenness seamless"
ZERO_INDEXED=""

# Parse arguments
MODEL=""
INSTANCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-attacks)
            N_ATTACKS="$2"
            shift 2
            ;;
        --m-min)
            M_MIN="$2"
            shift 2
            ;;
        --m-max)
            M_MAX="$2"
            shift 2
            ;;
        --protocols)
            PROTOCOLS="$2"
            shift 2
            ;;
        --zero-indexed)
            ZERO_INDEXED="--zero-indexed"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 MODEL INSTANCE [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  MODEL      Benchmark model (BA, ER, WS, SF2, LFR, etc.)"
            echo "  INSTANCE   Instance number (1-100)"
            echo ""
            echo "Options:"
            echo "  --n-attacks N    Number of stochastic replicates (default: 10)"
            echo "  --m-min M        Minimum m for SEAMLESS (default: 1)"
            echo "  --m-max M        Maximum m for SEAMLESS (default: 20)"
            echo "  --protocols P    Space-separated list of protocols"
            echo "  --zero-indexed   Use if network uses 0-indexed nodes"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            elif [[ -z "$INSTANCE" ]]; then
                INSTANCE="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$MODEL" ]]; then
    echo "Error: MODEL is required"
    echo "Usage: $0 MODEL INSTANCE"
    exit 1
fi

if [[ -z "$INSTANCE" ]]; then
    echo "Error: INSTANCE is required"
    echo "Usage: $0 MODEL INSTANCE"
    exit 1
fi

# Paths
BENCHMARK_DIR="benchmarks"
#EDGE_FILE="${BENCHMARK_DIR}/${MODEL}_N256/${MODEL}_r${INSTANCE}.edges"

#RJ: EDGE_FILE=$(ls ${BENCHMARK_DIR}/${MODEL}_N256/${MODEL}*_r${INSTANCE}.edges)

EDGE_FILE=$(ls ${BENCHMARK_DIR}/${MODEL}/*.edges | sort -V | sed -n "${INSTANCE}p")
#OUTPUT_DIR="results_${MODEL}_r${INSTANCE}"

#OUTPUT_DIR="results_$(basename "$EDGE_FILE" .edges)"

FILE_NAME=$(basename "$EDGE_FILE" .edges)
LABEL="${MODEL}_${FILE_NAME}"
OUTPUT_DIR="results_${LABEL}"

# Check if edge file exists
if [[ ! -f "$EDGE_FILE" ]]; then
    echo "Error: Edge file not found: $EDGE_FILE"
    echo "Available models in $BENCHMARK_DIR:"
    ls -1 "$BENCHMARK_DIR" | grep "_N256" | sed 's/_N256//' | sort
    exit 1
fi



echo "============================================================"
echo "SEAMLESS Analysis Runner"
echo "============================================================"
echo "Model:      $MODEL"
echo "Instance:   $INSTANCE"
echo "Edge file:  $EDGE_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "N attacks:  $N_ATTACKS"
echo "m range:    $M_MIN to $M_MAX"
echo "Protocols:  $PROTOCOLS"
echo "============================================================"
echo ""


# Run Python analysis
echo "Running SEAMLESS robustness analysis..."
py seamless_robustness.py \
    --edge-file "$EDGE_FILE" \
    --label "$LABEL" \
    --outdir "$OUTPUT_DIR" \
    --n-attacks "$N_ATTACKS" \
    --m-min "$M_MIN" \
    --m-max "$M_MAX" \
    --protocols $PROTOCOLS \
    $ZERO_INDEXED

echo ""
echo "============================================================"
echo "Generating plots..."
echo "============================================================"

# Run R plotting script
#Rscript plot_seamless.R --dir "$OUTPUT_DIR"

#python plot_seamless.py --dir "$OUTPUT_DIR" 

echo ""
echo "============================================================"
echo "✓ COMPLETE"
echo "============================================================"
echo "Results:  $OUTPUT_DIR/"
echo "Plots:    $OUTPUT_DIR/plots.pdf"
echo ""
echo "Files generated:"
ls -1 "$OUTPUT_DIR/"
echo "============================================================"