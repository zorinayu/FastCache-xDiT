#!/bin/bash
# 直接使用FastCache的测试脚本

# 默认模型类型为SD3
MODEL_TYPE="sd3"
if [ "$1" == "flux" ]; then
    MODEL_TYPE="flux"
    MODEL="black-forest-labs/FLUX.1-schnell"
else
    MODEL="stabilityai/stable-diffusion-3-medium-diffusers"
fi

# 运行测试
echo "==================== FastCache 测试 ===================="
echo "模型类型: $MODEL_TYPE"
echo "模型: $MODEL"
echo "======================================================="

# 添加项目根目录到Python路径
export PYTHONPATH=$PWD:$PYTHONPATH

# 运行测试脚本
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL

echo "======================================================="
echo "测试完成！" 