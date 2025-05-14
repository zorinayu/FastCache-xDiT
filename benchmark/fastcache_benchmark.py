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

# 导入缓存模块
try:
    # 导入所有缓存类
    from xfuser.model_executor.cache.utils import (
        FBCachedTransformerBlocks, 
        TeaCachedTransformerBlocks, 
        FastCachedTransformerBlocks,
        CacheContext
    )
    
    # 检查是否成功导入
    cache_modules_available = True
    print("Successfully imported cache modules from xfuser")
except ImportError as e:
    print(f"Warning: xfuser cache modules not found: {e}")
    cache_modules_available = False

def parse_args():
    parser = argparse.ArgumentParser(description="Cache Benchmark Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux", "pixart"], default="pixart")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_methods", type=str, nargs="+", 
                        choices=["None", "Fast", "Fb", "Tea"], 
                        default=["None", "Fast", "Fb", "Tea"])
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="cache_benchmark_results")
    
    args = parser.parse_args()
    
    # Set default model based on model_type if not provided
    if args.model is None:
        if args.model_type == "sd3":
            args.model = "stabilityai/stable-diffusion-3-medium-diffusers"
        elif args.model_type == "flux":
            args.model = "black-forest-labs/FLUX.1-schnell"
        elif args.model_type == "pixart":
            args.model = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    
    return args

def apply_cache(model, cache_method, cache_threshold=0.05, motion_threshold=0.1, num_steps=20):
    """应用指定的缓存方法到模型"""
    # 确认缓存模块可用
    if not cache_modules_available:
        print(f"Warning: Cache modules not available, skipping {cache_method}Cache application")
        return model, None
    
    # 查找并应用缓存到transformer块
    if not hasattr(model, "transformer"):
        print(f"WARNING: Model does not have a transformer attribute, cannot apply {cache_method}Cache")
        return model, None
    
    transformer = model.transformer
    
    # 检查是否有transformer_blocks属性
    if not hasattr(transformer, "transformer_blocks"):
        print(f"WARNING: Transformer does not have transformer_blocks, cannot apply {cache_method}Cache")
        return model, None
    
    # 获取transformer块和single_transformer_blocks (如果有的话)
    transformer_blocks = transformer.transformer_blocks
    single_transformer_blocks = None
    if hasattr(transformer, "single_transformer_blocks"):
        single_transformer_blocks = transformer.single_transformer_blocks
    
    # 保存原始的transformer_blocks
    original_transformer_blocks = transformer_blocks
    original_single_transformer_blocks = single_transformer_blocks
    
    # 为了记录缓存命中统计
    stats_collector = {"hits": 0, "total": 0}
    
    # 根据缓存方法选择相应的类
    if cache_method == "Fb":
        # 创建FBCache
        cached_blocks = FBCachedTransformerBlocks(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=cache_threshold,
            return_hidden_states_first=False,
            num_steps=num_steps
        )
        print(f"Applied FBCache to transformer with threshold {cache_threshold}")
    
    elif cache_method == "Tea":
        # 创建TeaCache
        cached_blocks = TeaCachedTransformerBlocks(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=cache_threshold,
            return_hidden_states_first=False,
            num_steps=num_steps
        )
        print(f"Applied TeaCache to transformer with threshold {cache_threshold}")
    
    elif cache_method == "Fast":
        # 创建FastCache
        cached_blocks = FastCachedTransformerBlocks(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=cache_threshold,
            return_hidden_states_first=False,
            num_steps=num_steps,
            motion_threshold=motion_threshold
        )
        print(f"Applied FastCache to transformer with threshold {cache_threshold}, motion {motion_threshold}")
    
    else:
        print(f"Unknown cache method: {cache_method}")
        return model, None
    
    # 用缓存块替换原来的块
    dummy_single_blocks = torch.nn.ModuleList() if single_transformer_blocks else None
    
    # 保存原始的forward函数
    original_forward = transformer.forward
    
    # 创建新的forward函数
    def new_forward(self, *args, **kwargs):
        # 暂时替换transformer_blocks
        temp_blocks = self.transformer_blocks
        temp_single_blocks = self.single_transformer_blocks if hasattr(self, "single_transformer_blocks") else None
        
        # 设置缓存块
        self.transformer_blocks = cached_blocks
        if hasattr(self, "single_transformer_blocks"):
            self.single_transformer_blocks = dummy_single_blocks
        
        # 调用原始forward
        try:
            result = original_forward(*args, **kwargs)
        finally:
            # 恢复原始块
            self.transformer_blocks = temp_blocks
            if hasattr(self, "single_transformer_blocks"):
                self.single_transformer_blocks = temp_single_blocks
        
        # 收集统计数据
        if hasattr(cached_blocks, "cnt"):
            stats_collector["total"] = int(cached_blocks.cnt.item())
        if hasattr(cached_blocks, "use_cache"):
            stats_collector["hits"] = int(cached_blocks.use_cache.sum().item())
        
        return result
    
    # 替换forward函数
    transformer.forward = new_forward.__get__(transformer)
    
    return model, stats_collector

def get_model_for_method(args, method, original_model=None):
    """获取应用了指定缓存方法的模型实例"""
    if original_model is not None and method == "None":
        # 对于基线，返回原始模型
        return original_model, None
    
    # 加载一个新的模型实例
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
    stats_collector = None
    
    if method == "None":
        # 不使用缓存 - 用作基线
        pass
    
    elif method in ["Fb", "Tea", "Fast"]:
        # 应用缓存方法
        model, stats_collector = apply_cache(
            model,
            cache_method=method,
            cache_threshold=args.cache_ratio_threshold,
            motion_threshold=args.motion_threshold,
            num_steps=args.num_inference_steps
        )
    
    return model, stats_collector

def main():
    args = parse_args()
    print(f"Benchmarking cache methods for {args.model} ({args.model_type})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load baseline model
    baseline_model = None
    if "None" in args.cache_methods:
        if args.model_type == "sd3":
            baseline_model = StableDiffusion3Pipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "flux":
            baseline_model = FluxPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "pixart":
            baseline_model = PixArtSigmaPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
    
    # Benchmark results
    results = {}
    
    # Run benchmarks for each cache method
    for method in args.cache_methods:
        print(f"\n===== Testing {method}Cache =====")
        method_name = "Baseline" if method == "None" else f"{method}Cache"
        
        # Get model with cache method applied
        model, stats_collector = get_model_for_method(args, method, baseline_model)
        
        # Run inference with timing
        print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
        start_time = time.time()
        
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            result = model(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                generator=generator,
            )
        
        inference_time = time.time() - start_time
        print(f"{method_name} inference completed in {inference_time:.2f} seconds")
        
        # Save generated image
        output_image = result.images[0]
        image_path = os.path.join(args.output_dir, f"{method.lower()}_image.png")
        output_image.save(image_path)
        print(f"Image saved to {image_path}")
        
        # Collect statistics
        stats = {
            "method": method_name,
            "model": args.model,
            "model_type": args.model_type,
            "prompt": args.prompt,
            "steps": args.num_inference_steps,
            "resolution": f"{args.height}x{args.width}",
            "seed": args.seed,
            "inference_time": inference_time,
        }
        
        # Add cache-specific stats
        if method != "None" and stats_collector:
            hits = stats_collector.get("hits", 0)
            total = stats_collector.get("total", 0)
            hit_ratio = hits / total if total > 0 else 0
            
            stats["cache_stats"] = {
                "hits": hits,
                "total": total,
                "hit_ratio": hit_ratio
            }
            
            stats["cache_threshold"] = args.cache_ratio_threshold
            if method == "Fast":
                stats["motion_threshold"] = args.motion_threshold
            
            if total > 0:
                print(f"\nCache hit statistics:")
                print(f"Hits: {hits}/{total} ({hit_ratio:.2%})")
        
        # Store results
        results[method] = stats
    
    # Save overall benchmark results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to {results_path}")
    
    # Print comparison summary
    baseline_time = results.get("None", {}).get("inference_time", 0)
    if baseline_time > 0:
        print("\n===== Cache Methods Comparison =====")
        print(f"{'Method':<15} {'Time (s)':<10} {'Speedup':<10}")
        print("-" * 35)
        
        for method in args.cache_methods:
            method_time = results[method]["inference_time"]
            speedup = baseline_time / method_time if method != "None" else 1.0
            print(f"{results[method]['method']:<15} {method_time:<10.2f} {speedup:<10.2f}x")
    
if __name__ == "__main__":
    main() 