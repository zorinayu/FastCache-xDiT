#!/bin/bash
# Run PixArt FastCache benchmark

# Create output directory
mkdir -p cache_results

# Run benchmark with different cache methods
python benchmark/cache_execute.py \
  --model_type pixart \
  --model "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
  --prompt "a colorful landscape with mountains and a serene lake" \
  --cache_methods None Fast Fb Tea \
  --num_inference_steps 30 \
  --height 512 \
  --width 512 \
  --cache_ratio_threshold 0.05 \
  --motion_threshold 0.1 \
  --output_dir cache_results

echo "Benchmark completed! Results saved to cache_results/" 