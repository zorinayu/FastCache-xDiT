#!/bin/bash
# Run FastCache benchmark comparison

# Default model is SD3
MODEL_TYPE="sd3"
if [ "$1" == "flux" ]; then
    MODEL_TYPE="flux"
    MODEL="black-forest-labs/FLUX.1-schnell"
else
    MODEL="stabilityai/stable-diffusion-3-medium-diffusers"
fi

# Run benchmark
python examples/fastcache_benchmark.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "a beautiful landscape with mountains and a lake at sunset" \
    --num_inference_steps 30 \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1 \
    --repeat 3 \
    --output_dir "fastcache_benchmark_results_${MODEL_TYPE}"

echo "Benchmark results saved to fastcache_benchmark_results_${MODEL_TYPE}/" 