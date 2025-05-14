#!/bin/bash
# FastCache Benchmark: Compare FastCache with FBCache and TeaCache

# Set model type based on parameter (default to Flux)
MODEL_TYPE="flux"
MODEL="black-forest-labs/FLUX.1-schnell"
PROMPT="a serene landscape with mountains and a lake"
NUM_STEPS=30
OUTPUT_DIR="fastcache_benchmark_results"

# Parse command-line arguments
if [ "$1" == "pixart" ]; then
    MODEL_TYPE="pixart"
    MODEL="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    PROMPT="a photo of an astronaut riding a horse on the moon"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Display benchmark parameters
echo "==================== FastCache Benchmark ===================="
echo "Model type: $MODEL_TYPE"
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Steps: $NUM_STEPS"
echo "=========================================================="

# Add project root directory to Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Run baseline (no cache)
echo "Running baseline (no cache)..."
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --cache_method "None" \
    --output_dir "$OUTPUT_DIR"

# Run FastCache
echo "Running FastCache..."
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --cache_method "Fast" \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1 \
    --output_dir "$OUTPUT_DIR"

# Run FBCache (First Block Cache)
echo "Running FBCache..."
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --cache_method "Fb" \
    --cache_ratio_threshold 0.6 \
    --output_dir "$OUTPUT_DIR"

# Run TeaCache
echo "Running TeaCache..."
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --cache_method "Tea" \
    --cache_ratio_threshold 0.6 \
    --output_dir "$OUTPUT_DIR"

# Generate comparison report
echo "Generating comparison report..."
python examples/generate_benchmark_report.py \
    --output_dir "$OUTPUT_DIR" \
    --model_type $MODEL_TYPE

echo "=========================================================="
echo "Benchmark complete! Results saved to $OUTPUT_DIR/" 