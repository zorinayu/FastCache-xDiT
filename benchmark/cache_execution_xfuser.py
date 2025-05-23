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

# Import diffusers first
try:
    from diffusers import PixArtSigmaPipeline, StableDiffusion3Pipeline, FluxPipeline
    diffusers_available = True
except ImportError as e:
    print(f"Warning: diffusers not available: {e}")
    diffusers_available = False

# Import xfuser components
try:
    from xfuser import (
        xFuserPixArtSigmaPipeline, 
        xFuserStableDiffusion3Pipeline, 
        xFuserFluxPipeline,
        xFuserArgs
    )
    from xfuser.config import FlexibleArgumentParser
    from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
    xfuser_available = True
    print("Successfully imported xfuser modules")
except ImportError as e:
    print(f"Warning: xfuser modules not found: {e}")
    xfuser_available = False

def parse_args():
    parser = argparse.ArgumentParser(description="xFuser Cache Execution Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux", "pixart"], default="pixart")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on the moon")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_methods", type=str, nargs="+", 
                        choices=["None", "Fast", "Fb", "Tea", "All"], 
                        default=["All"])
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="cache_xfuser_results")
    
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

def load_baseline_model(model_type, model_name):
    """加载基础diffusers模型（不使用缓存）"""
    if not diffusers_available:
        raise ImportError("diffusers not available")
    
    print(f"Loading baseline {model_type} model: {model_name}")
    
    if model_type == "pixart":
        model = PixArtSigmaPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
    elif model_type == "flux":
        model = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
    elif model_type == "sd3":
        model = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def load_fastcache_model(model_type, model_name, cache_ratio_threshold, motion_threshold):
    """加载带FastCache的模型"""
    if not diffusers_available:
        raise ImportError("diffusers not available")
    
    print(f"Loading {model_type} model with FastCache: {model_name}")
    
    # 首先加载基础模型
    base_model = load_baseline_model(model_type, model_name)
    
    # 用FastCache包装器包装
    fastcache_wrapper = xFuserFastCachePipelineWrapper(base_model)
    
    # 启用FastCache
    fastcache_wrapper.enable_fastcache(
        cache_ratio_threshold=cache_ratio_threshold,
        motion_threshold=motion_threshold,
    )
    
    print(f"FastCache enabled with threshold={cache_ratio_threshold}, motion={motion_threshold}")
    return fastcache_wrapper

def load_xfuser_cache_model(model_type, model_name, cache_method, num_steps):
    """加载带xfuser缓存的模型（Fb, Tea）"""
    if not xfuser_available:
        raise ImportError("xfuser not available")
    
    print(f"Loading {model_type} model with {cache_method}Cache: {model_name}")
    
    # 创建 engine_args
    engine_args_list = [
        "--model", model_name,
        "--prompt", "dummy prompt",  # 临时的，会被实际prompt覆盖
        "--num_inference_steps", str(num_steps)
    ]
    
    # 根据cache_method添加相应参数
    if cache_method == "Fb":
        engine_args_list.extend(["--use_fbcache"])
    elif cache_method == "Tea":
        engine_args_list.extend(["--use_teacache"])
    
    # 解析参数
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    xfuser_args = xFuserArgs.add_cli_args(parser).parse_args(engine_args_list)
    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    
    # 设置cache参数
    cache_args = {
        "use_teacache": cache_method == "Tea",
        "use_fbcache": cache_method == "Fb",
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": num_steps,
    }
    
    # 加载对应的pipeline
    if model_type == "pixart":
        pipe = xFuserPixArtSigmaPipeline.from_pretrained(
            pretrained_model_name_or_path=model_name,
            engine_config=engine_config,
            cache_args=cache_args,
            torch_dtype=torch.float16,
        )
    elif model_type == "flux":
        pipe = xFuserFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=model_name,
            engine_config=engine_config,
            cache_args=cache_args,
            torch_dtype=torch.float16,
        )
    elif model_type == "sd3":
        pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=model_name,
            engine_config=engine_config,
            cache_args=cache_args,
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 设置设备
    pipe = pipe.to("cuda")
    
    return pipe, input_config

def run_baseline_inference(model, args, generator):
    """运行基础推理（无缓存）"""
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

def run_fastcache_inference(fastcache_wrapper, args, generator):
    """运行FastCache推理"""
    start_time = time.time()
    
    with torch.no_grad():
        result = fastcache_wrapper(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            generator=generator,
        )
    
    inference_time = time.time() - start_time
    
    # 获取缓存统计信息
    cache_stats = fastcache_wrapper.get_cache_statistics()
    return result, inference_time, cache_stats

def run_xfuser_inference(pipe, input_config, args, generator):
    """运行xfuser推理（Fb, Tea）"""
    start_time = time.time()
    
    # 更新input_config
    input_config.prompt = [args.prompt]
    input_config.height = args.height
    input_config.width = args.width
    input_config.num_inference_steps = args.num_inference_steps
    input_config.seed = args.seed
    
    # 准备运行
    pipe.prepare_run(input_config, steps=args.num_inference_steps)
    
    with torch.no_grad():
        result = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
    
    inference_time = time.time() - start_time
    return result, inference_time

def main():
    args = parse_args()
    print(f"Testing cache methods for {args.model} ({args.model_type})")
    
    if not diffusers_available:
        print("diffusers is not available. Please install diffusers first.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 存储所有结果
    results = {}
    
    # 为每个缓存方法运行测试
    for method in args.cache_methods:
        print(f"\n===== Testing {method}Cache =====")
        method_name = "Baseline" if method == "None" else f"{method}Cache"
        
        try:
            # 创建固定的随机生成器以确保结果一致
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            
            # 根据方法类型加载不同的模型和运行推理
            if method == "None":
                # 基础模型（无缓存）
                model = load_baseline_model(args.model_type, args.model)
                print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
                result, inference_time = run_baseline_inference(model, args, generator)
                cache_stats = None
                
            elif method == "Fast":
                # FastCache模型
                fastcache_wrapper = load_fastcache_model(
                    args.model_type, 
                    args.model, 
                    args.cache_ratio_threshold, 
                    args.motion_threshold
                )
                print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
                result, inference_time, cache_stats = run_fastcache_inference(fastcache_wrapper, args, generator)
                
            elif method in ["Fb", "Tea"]:
                # xfuser缓存模型
                if not xfuser_available:
                    print(f"xfuser not available, skipping {method}Cache")
                    continue
                    
                pipe, input_config = load_xfuser_cache_model(
                    args.model_type, 
                    args.model, 
                    method, 
                    args.num_inference_steps
                )
                print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
                result, inference_time = run_xfuser_inference(pipe, input_config, args, generator)
                cache_stats = None
            
            print(f"{method_name} inference completed in {inference_time:.2f} seconds")
            
            # 保存生成的图像
            if hasattr(result, 'images') and len(result.images) > 0:
                output_image = result.images[0]
                image_path = os.path.join(args.output_dir, f"{method.lower()}_image.png")
                output_image.save(image_path)
                print(f"Image saved to {image_path}")
            else:
                print("Warning: No images in result")
            
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
            
            # 添加FastCache特有的参数和统计信息
            if method == "Fast":
                results[method]["motion_threshold"] = args.motion_threshold
                if cache_stats:
                    results[method]["cache_stats"] = cache_stats
                    print(f"FastCache statistics:")
                    for name, stats in cache_stats.items():
                        hit_ratio = stats.get('cache_hit_ratio', 0)
                        print(f"  {name}: {hit_ratio:.2%} hit ratio")
                
        except Exception as e:
            print(f"Error during {method_name} test: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存整体测试结果
    results_path = os.path.join(args.output_dir, "cache_execution_results.json")
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
            if method in results:
                method_time = results[method]["inference_time"]
                speedup = baseline_time / method_time if method != "None" else 1.0
                print(f"{results[method]['method']:<15} {method_time:<10.2f} {speedup:<10.2f}x")
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 