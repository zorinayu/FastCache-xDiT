#!/bin/bash
# 一键运行所有缓存方法的对比测试

# 设置默认参数
MODEL_TYPE=${1:-"pixart"}  # 可选: pixart, flux, sd3
STEPS=${2:-20}
HEIGHT=${3:-512}
WIDTH=${4:-512}
OUTPUT_DIR="cache_comparison_results/${MODEL_TYPE}"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "Running cache comparison for $MODEL_TYPE models with $STEPS steps at ${HEIGHT}x${WIDTH} resolution"

# 运行所有缓存方法的对比测试
python benchmark/cache_execute.py \
  --model_type $MODEL_TYPE \
  --cache_methods All \
  --num_inference_steps $STEPS \
  --height $HEIGHT \
  --width $WIDTH \
  --output_dir $OUTPUT_DIR

echo "Comparison complete!"
echo "Results saved to $OUTPUT_DIR"
echo "You can examine the JSON report and generated images to compare performance and quality." 