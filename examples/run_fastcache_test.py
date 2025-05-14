import os
import sys
import time
import torch
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 尝试导入必要的模块
try:
    from diffusers import StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Warning: diffusers库未正确安装，请确保安装了diffusers>=0.30.0")

from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator

def parse_args():
    parser = argparse.ArgumentParser(description="FastCache Simple Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux"], default="sd3")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake")
    return parser.parse_args()

def create_simple_accelerator(module, cache_threshold=0.05, motion_threshold=0.1):
    """创建一个简单的FastCache加速器，不依赖注册机制"""
    accelerator = FastCacheAccelerator(
        module, 
        cache_ratio_threshold=cache_threshold,
        motion_threshold=motion_threshold
    )
    return accelerator

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
            
        # 如果模型加载成功，先运行基准测试
        print("Model loaded successfully. Running baseline inference...")
        start_time = time.time()
        
        with torch.no_grad():
            image = model(
                prompt=args.prompt,
                num_inference_steps=5,  # 使用较少的步骤用于测试
                height=512,
                width=512,
            ).images[0]
            
        baseline_time = time.time() - start_time
        print(f"Baseline inference completed in {baseline_time:.2f} seconds")
        
        # 创建简单的FastCache加速模块
        print("\nAdding FastCache accelerator to transformer blocks...")
        
        # 找到transformer模块并应用FastCache
        accelerators = []
        if hasattr(model, "unet"):
            transformer = model.unet
            
            # 递归查找并应用FastCache到transformer块
            def apply_fastcache(module, prefix=''):
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name
                    
                    # 检查模块类型
                    module_type = child.__class__.__name__
                    if "Transformer" in module_type or "Attention" in module_type:
                        # 创建加速器
                        accelerator = create_simple_accelerator(child)
                        accelerators.append((full_name, accelerator))
                        
                        # 替换原始模块
                        setattr(module, name, accelerator)
                        print(f"Applied FastCache to {module_type} at {full_name}")
                    else:
                        # 递归处理子模块
                        apply_fastcache(child, full_name)
            
            apply_fastcache(transformer)
            
            print(f"Applied FastCache to {len(accelerators)} transformer blocks")
            
            # 使用FastCache运行推理
            print("\nRunning inference with FastCache...")
            start_time = time.time()
            
            with torch.no_grad():
                image = model(
                    prompt=args.prompt,
                    num_inference_steps=5,  # 使用较少的步骤用于测试
                    height=512,
                    width=512,
                )
                
            fastcache_time = time.time() - start_time
            print(f"FastCache inference completed in {fastcache_time:.2f} seconds")
            
            # 显示加速结果
            if fastcache_time > 0 and baseline_time > 0:
                speedup = baseline_time / fastcache_time
                print(f"\nFastCache speedup: {speedup:.2f}x")
                
                # 显示缓存命中率
                print("\nCache hit statistics:")
                total_hits = 0
                total_steps = 0
                for name, acc in accelerators:
                    hits = acc.cache_hits
                    steps = acc.total_steps
                    total_hits += hits
                    total_steps += steps
                    if steps > 0:
                        print(f"  {name}: {hits}/{steps} hits ({hits/steps:.2%})")
                
                if total_steps > 0:
                    print(f"Overall: {total_hits}/{total_steps} hits ({total_hits/total_steps:.2%})")
        else:
            print("Could not find transformer blocks in the model. FastCache not applied.")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 