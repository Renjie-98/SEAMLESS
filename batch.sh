#!/bin/bash
#
# batch.sh
# Run SEAMLESS analysis on all benchmark networks
#
# Usage:
#   ./batch.sh              # Default: models in benchmarks/, instances 1-5
#   ./batch.sh 1 10         # Custom instance range: 1-10
#   ./batch.sh 3 8 BA ER           # Only BA and ER models, instances 3-8
#

set -e

# Default parameters
INSTANCE_MIN=
INSTANCE_MAX=
MODELS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [INSTANCE_MIN INSTANCE_MAX] [MODELS...]"
            echo ""
            echo "Arguments:"
            echo "  INSTANCE_MIN   First instance number (default: 1)"
            echo "  INSTANCE_MAX   Last instance number (default: 5)"
            echo "  MODELS         Specific models to run (default: all in benchmarks/)"
            echo ""
            echo "Examples:"
            echo "  $0                          # All models, instances 1-5"
            echo "  $0 1 10                     # All models, instances 1-10"
            echo "  $0 1 5 BA ER WS             # Only BA, ER, WS models"
            echo "  $0 3 7 LFR                  # Only LFR, instances 3-7"
            exit 0
            ;;
        *)
            if [[ -z "$INSTANCE_MIN" ]] || [[ "$INSTANCE_MIN" == "$1" ]]; then
                INSTANCE_MIN="$1"
            elif [[ -z "$INSTANCE_MAX" ]] || [[ "$INSTANCE_MAX" == "$1" ]]; then
                INSTANCE_MAX="$1"
            else
                MODELS="$MODELS $1"
            fi
            shift
            ;;
    esac
done

# If no models specified, get all from benchmarks/
#if [[ -z "$MODELS" ]]; then
   # MODELS=$(ls -1 benchmarks/ | grep "_N256$" | sed 's/_N256//' | sort)
#fi

echo "============================================================"
echo "SEAMLESS Batch Runner"
echo "============================================================"
echo "Models:     $MODELS"
echo "Instances:  $INSTANCE_MIN to $INSTANCE_MAX"
echo "Total runs: $(echo $MODELS | wc -w) models × $((INSTANCE_MAX - INSTANCE_MIN + 1)) instances = $(($(echo $MODELS | wc -w) * (INSTANCE_MAX - INSTANCE_MIN + 1)))"
echo "============================================================"
echo ""

# Count total operations
TOTAL=0
for MODEL in $MODELS; do
    for INSTANCE in $(seq $INSTANCE_MIN $INSTANCE_MAX); do
        TOTAL=$((TOTAL + 1))
    done
done

CURRENT=0
START_TIME=$(date +%s)

#RJ new begin:
MAX_JOBS=24 
#RJ new end

for MODEL in $MODELS; do
    for INSTANCE in $(seq $INSTANCE_MIN $INSTANCE_MAX); do
        CURRENT=$((CURRENT + 1))
        
        # Calculate elapsed and estimated remaining time
        ELAPSED=$(($(date +%s) - START_TIME))
        if [[ $CURRENT -gt 1 ]]; then
            AVG_TIME=$((ELAPSED / (CURRENT - 1)))
            REMAINING=$((AVG_TIME * (TOTAL - CURRENT + 1)))
            REMAINING_STR=" (ETA: ${REMAINING}s)"
        else
            REMAINING_STR=""
        fi
        
        echo "------------------------------------------------------------"
        echo "[$CURRENT/$TOTAL] Running $MODEL r$INSTANCE...$REMAINING_STR"
        echo "------------------------------------------------------------"

        if [[ "$MODEL" == "L" && "$INSTANCE" -gt 1 ]]; then
            echo "Skipping L r$INSTANCE: only one .edges file available"
            echo ""
            continue
        fi
        # RJ old begin:
        #./run_seamless.sh "$MODEL" "$INSTANCE" --zero-indexed
        # RJ old end:

        # RJ new begin Serial:
        # ./run_seamless.sh "$MODEL" "$INSTANCE" --protocols "seamless" --m-min 1 --m-max 6 --zero-indexed
        # echo ""        
        # RJ new end:
        
        # RJ new begin parallel:
        # Limit the number of concurrent jobs
        while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 1
        done
        #./run_seamless.sh "$MODEL" "$INSTANCE" --protocols "seamless" --m-min 4 --m-max 4  &
        ./run_seamless.sh "$MODEL" "$INSTANCE" --protocols "random degree adap_degree betweenness adap_betweenness seamless" --m-min 4 --m-max 4 --zero-indexed  &
        echo "Running jobs: $(jobs -r | wc -l)"        
        # RJ new end
        
    done
done

# RJ new begin:
wait
# RJ new end:

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "============================================================"
echo "✓ ALL COMPLETE"
echo "============================================================"
echo "Total time: ${TOTAL_TIME}s"
echo "Results directories:"
for MODEL in $MODELS; do
    for INSTANCE in $(seq $INSTANCE_MIN $INSTANCE_MAX); do
        EDGE_FILE=$(ls benchmarks/${MODEL}/*.edges | sort -V | sed -n "${INSTANCE}p")
        FILE_NAME=$(basename "$EDGE_FILE" .edges)
        echo "  results_${MODEL}_${FILE_NAME}/" 
    done
done
echo "============================================================"