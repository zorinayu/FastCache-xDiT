import os
import sys
import time
import torch
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Try to import necessary modules
try:
    from diffusers import PixArtSigmaPipeline, StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Warning: diffusers library not properly installed, make sure you have diffusers>=0.30.0")

# 导入cache相关函数
try:
    # 直接从flux.py导入缓存应用函数
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
    cache_modules_available = True
    print("Successfully imported cache modules from xfuser")
except ImportError as e:
    print(f"Warning: xfuser cache modules not found: {e}")
    cache_modules_available = False

def parse_args():
    parser = argparse.ArgumentParser(description="Cache Execution Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux", "pixart"], default="pixart")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_methods", type=str, nargs="+", 
                        choices=["None", "Fast", "Fb", "Tea", "All"], 
                        default=["All"])
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="cache_execute_results")
    
    args = parser.parse_args()
    
    # Set default model based on model_type if not provided
    if args.model is None:
        if args.model_type == "sd3":
            args.model = "stabilityai/stable-diffusion-3-medium-diffusers"
        elif args.model_type == "flux":
            args.model = "black-forest-labs/FLUX.1-schnell"
        elif args.model_type == "pixart":
            args.model = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    
    # 如果选择了All，则测试所有缓存方法
    if "All" in args.cache_methods:
        args.cache_methods = ["None", "Fast", "Fb", "Tea"]
    
    return args

def apply_cache_to_model(model, cache_method, rel_l1_thresh=0.05, num_steps=20, motion_threshold=0.1):
    """直接使用apply_cache_on_transformer函数应用缓存方法"""
    if not cache_modules_available:
        print(f"Warning: Cache modules not available, cannot apply {cache_method}")
        return model
    
    if not hasattr(model, "transformer"):
        print(f"Warning: Model does not have a transformer attribute, cannot apply {cache_method}")
        return model
    
    transformer = model.transformer
    
    if cache_method == "None":
        # 不应用缓存
        return model
    
    try:
        # 直接使用flux.py中的函数
        apply_cache_on_transformer(
            transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=False,
            num_steps=num_steps,
            use_cache=cache_method,  # "Fast", "Fb", or "Tea"
            motion_threshold=motion_threshold
        )
        print(f"Successfully applied {cache_method}Cache to transformer")
    except Exception as e:
        print(f"Error applying {cache_method}Cache: {e}")
    
    return model

def run_inference(model, args, generator):
    """运行推理并计时"""
    start_time = time.time()
    
    with torch.no_grad():
        result = model(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            generator=generator,
        )
    
    inference_time = time.time() - start_time
    
    return result, inference_time

def main():
    args = parse_args()
    print(f"Testing cache methods for {args.model} ({args.model_type})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 存储所有结果
    results = {}
    
    # 为每个缓存方法加载新的模型实例并运行推理
    for method in args.cache_methods:
        print(f"\n===== Testing {method}Cache =====")
        method_name = "Baseline" if method == "None" else f"{method}Cache"
        
        # 加载模型
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
        elif args.model_type == "pixart":
            model = PixArtSigmaPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        
        # 应用缓存方法
        model = apply_cache_to_model(
            model,
            cache_method=method,
            rel_l1_thresh=args.cache_ratio_threshold,
            num_steps=args.num_inference_steps,
            motion_threshold=args.motion_threshold
        )
        
        # 创建固定的随机生成器以确保结果一致
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        
        # 运行推理
        print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
        result, inference_time = run_inference(model, args, generator)
        print(f"{method_name} inference completed in {inference_time:.2f} seconds")
        
        # 保存生成的图像
        output_image = result.images[0]
        image_path = os.path.join(args.output_dir, f"{method.lower()}_image.png")
        output_image.save(image_path)
        print(f"Image saved to {image_path}")
        
        # 收集统计信息
        results[method] = {
            "method": method_name,
            "model": args.model,
            "model_type": args.model_type,
            "prompt": args.prompt,
            "steps": args.num_inference_steps,
            "resolution": f"{args.height}x{args.width}",
            "seed": args.seed,
            "inference_time": inference_time,
            "cache_threshold": args.cache_ratio_threshold
        }
        
        # 为FastCache方法添加motion_threshold
        if method == "Fast":
            results[method]["motion_threshold"] = args.motion_threshold
    
    # 保存整体测试结果
    results_path = os.path.join(args.output_dir, "cache_execute_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to {results_path}")
    
    # 打印比较摘要
    baseline_time = results.get("None", {}).get("inference_time", 0)
    if baseline_time > 0:
        print("\n===== Cache Methods Comparison =====")
        print(f"{'Method':<15} {'Time (s)':<10} {'Speedup':<10}")
        print("-" * 35)
        
        for method in args.cache_methods:
            method_time = results[method]["inference_time"]
            speedup = baseline_time / method_time if method != "None" else 1.0
            print(f"{results[method]['method']:<15} {method_time:<10.2f} {speedup:<10.2f}x")

# 仅用于测试DiTFastAttn的帮助函数（未完全实现）
def test_ditfastattn():
    """
    DiTFastAttn是一种注意力计算优化方法，而不是直接的缓存方法
    这个函数仅用于展示，实际上需要特定实现
    """
    print("DiTFastAttn is an attention computation optimization rather than a caching method.")
    print("It requires specific implementation for the model's attention mechanism.")
    print("Check README.md or docs/methods/ditfastattn.md for more information.")

if __name__ == "__main__":
    main() 