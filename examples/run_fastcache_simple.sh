#!/bin/bash
# 简化版FastCache基准测试脚本

# 默认模型类型为SD3
MODEL_TYPE="sd3"
if [ "$1" == "flux" ]; then
    MODEL_TYPE="flux"
    MODEL="black-forest-labs/FLUX.1-schnell"
else
    MODEL="stabilityai/stable-diffusion-3-medium-diffusers"
fi

# 运行简化版基准测试
echo "正在运行FastCache基准测试 (简化版)..."
echo "模型类型: $MODEL_TYPE"
echo "模型: $MODEL"

# 运行测试
python examples/fastcache_benchmark_simple.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "a beautiful landscape with mountains and a lake at sunset" \
    --num_inference_steps 30 \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1 \
    --output_dir "fastcache_results_${MODEL_TYPE}"

echo "结果已保存到 fastcache_results_${MODEL_TYPE}/" 