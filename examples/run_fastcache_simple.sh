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

# 添加项目根目录到Python路径
export PYTHONPATH=$PWD:$PYTHONPATH

# 创建简单的测试脚本
cat > examples/run_fastcache_test.py << 'EOL'
import os
import sys
import time
import torch
from pathlib import Path
import argparse

# 添加必要的import
try:
    from diffusers import StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Warning: diffusers库未正确安装，请确保安装了diffusers>=0.30.0")

def parse_args():
    parser = argparse.ArgumentParser(description="FastCache Simple Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux"], default="sd3")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing with model: {args.model} (type: {args.model_type})")
    
    # 加载模型
    try:
        if args.model_type == "sd3":
            model = StableDiffusion3Pipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "flux":
            model = FluxPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
            
        # 如果模型加载成功，直接生成一张图像
        print("Model loaded successfully. Generating an image...")
        start_time = time.time()
        
        with torch.no_grad():
            image = model(
                prompt=args.prompt,
                num_inference_steps=5,  # 使用较少的步骤用于测试
                height=512,
                width=512,
            ).images[0]
            
        end_time = time.time()
        print(f"Basic generation completed in {end_time - start_time:.2f} seconds")
        
        # 下一步将实现FastCache加速
        print("Base model working correctly. Now you can implement FastCache acceleration.")
        
    except Exception as e:
        print(f"Error during model loading or generation: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()
EOL

# 运行测试脚本
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL 