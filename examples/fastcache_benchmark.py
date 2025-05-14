import argparse
import os
import time
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from xfuser import xFuserArgs
from xfuser.parallel import xDiTParallel
from xfuser.config import FlexibleArgumentParser
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="FastCache Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Model path or name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sd3", "flux"],
        default="sd3",
        help="Model type (sd3 or flux)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on the moon",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--height", type=int, default=768, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=768, help="Image width"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fastcache_benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache_ratio_threshold",
        type=float,
        default=0.05,
        help="FastCache ratio threshold",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.1,
        help="FastCache motion threshold",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to repeat each benchmark",
    )
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(args):
    """Load the diffusion model based on model type"""
    print(f"Loading {args.model_type} model: {args.model}")
    
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
        
    return model


def run_baseline(model, args):
    """Run the model without any acceleration"""
    set_seed(args.seed)
    
    start_time = time.time()
    result = model(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def run_with_fastcache(model, args):
    """Run the model with FastCache acceleration"""
    set_seed(args.seed)
    
    # Create and configure FastCache wrapper
    fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
    fastcache_wrapper.enable_fastcache(
        cache_ratio_threshold=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
    )
    
    start_time = time.time()
    result = fastcache_wrapper(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    # Get cache statistics
    cache_stats = fastcache_wrapper.get_cache_statistics()
    print("FastCache statistics:")
    for name, stats in cache_stats.items():
        print(f"  {name}: {stats['cache_hit_ratio']:.2%} hit ratio")
    
    return result, end_time - start_time


def run_with_teacache(model, args):
    """Run the model with TeaCache acceleration using xFuserParallel API"""
    if args.model_type != "flux":
        print("TeaCache currently only supports Flux model")
        return None, float('inf')
    
    set_seed(args.seed)
    
    # Create xFuserArgs for TeaCache
    parser = FlexibleArgumentParser(description="xFuser TeaCache Args")
    flex_args = xFuserArgs.add_cli_args(parser).parse_args([
        "--model", args.model,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--use_teacache",  # Enable TeaCache
    ])
    engine_args = xFuserArgs.from_cli_args(flex_args)
    engine_config, input_config = engine_args.create_config()
    
    # Load model
    model = load_model(args)
    
    # Create xDiTParallel with TeaCache
    paralleler = xDiTParallel(model, engine_config, input_config)
    
    # Run inference
    start_time = time.time()
    result = paralleler(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def run_with_fbcache(model, args):
    """Run the model with First-Block-Cache acceleration using xFuserParallel API"""
    if args.model_type != "flux":
        print("First-Block-Cache currently only supports Flux model")
        return None, float('inf')
    
    set_seed(args.seed)
    
    # Create xFuserArgs for First-Block-Cache
    parser = FlexibleArgumentParser(description="xFuser FB Cache Args")
    flex_args = xFuserArgs.add_cli_args(parser).parse_args([
        "--model", args.model,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--use_fbcache",  # Enable First-Block-Cache
    ])
    engine_args = xFuserArgs.from_cli_args(flex_args)
    engine_config, input_config = engine_args.create_config()
    
    # Load model
    model = load_model(args)
    
    # Create xDiTParallel with First-Block-Cache
    paralleler = xDiTParallel(model, engine_config, input_config)
    
    # Run inference
    start_time = time.time()
    result = paralleler(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def save_results(results, times, args):
    """Save benchmark results and images"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save timing information
    timing_file = os.path.join(args.output_dir, "timing_results.txt")
    with open(timing_file, "w") as f:
        for method, time_list in times.items():
            avg_time = np.mean(time_list)
            f.write(f"{method}: {avg_time:.4f}s (avg of {len(time_list)} runs)\n")
    
    # Create bar chart of timings
    plt.figure(figsize=(10, 6))
    methods = []
    avg_times = []
    
    for method, time_list in times.items():
        if not time_list:  # Skip methods with no timing data
            continue
        methods.append(method)
        avg_times.append(np.mean(time_list))
    
    # Create bar chart
    colors = ['blue', 'green', 'orange', 'red']
    plt.bar(methods, avg_times, color=colors[:len(methods)])
    plt.ylabel('Time (seconds)')
    plt.title(f'Inference Time Comparison ({args.model_type})')
    
    # Add speedup labels
    baseline_time = times.get("Baseline", [0])[0]
    if baseline_time > 0:
        for i, time_val in enumerate(avg_times):
            if methods[i] != "Baseline":
                speedup = baseline_time / time_val
                plt.text(i, time_val, f"{speedup:.2f}x", ha='center', va='bottom')
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, "timing_comparison.png"))
    
    # Save images
    for method, result in results.items():
        if result is not None:
            image = result.images[0]
            image.save(os.path.join(args.output_dir, f"{method.lower()}_result.png"))


def main():
    args = parse_args()
    
    # Dictionary to store results and timings
    results = {}
    times = {
        "Baseline": [],
        "FastCache": [],
        "TeaCache": [],
        "FirstBlockCache": []
    }
    
    # Run benchmarks multiple times for more reliable results
    for i in range(args.repeat):
        print(f"\nBenchmark run {i+1}/{args.repeat}")
        
        # Load model (reload for each method to ensure fair comparison)
        model = load_model(args)
        
        # Run baseline
        print("\nRunning baseline...")
        result, elapsed = run_baseline(model, args)
        results["Baseline"] = result
        times["Baseline"].append(elapsed)
        print(f"Baseline completed in {elapsed:.4f}s")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Run with FastCache
        model = load_model(args)
        print("\nRunning with FastCache...")
        result, elapsed = run_with_fastcache(model, args)
        results["FastCache"] = result
        times["FastCache"].append(elapsed)
        print(f"FastCache completed in {elapsed:.4f}s")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Run with TeaCache for Flux models
        if args.model_type == "flux":
            print("\nRunning with TeaCache...")
            result, elapsed = run_with_teacache(model, args)
            if result is not None:
                results["TeaCache"] = result
                times["TeaCache"].append(elapsed)
                print(f"TeaCache completed in {elapsed:.4f}s")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Run with First-Block-Cache
            print("\nRunning with First-Block-Cache...")
            result, elapsed = run_with_fbcache(model, args)
            if result is not None:
                results["FirstBlockCache"] = result
                times["FirstBlockCache"].append(elapsed)
                print(f"First-Block-Cache completed in {elapsed:.4f}s")
    
    # Save results
    save_results(results, times, args)
    
    # Print final summary
    print("\n===== Benchmark Summary =====")
    for method, time_list in times.items():
        if time_list:  # Only report methods with timing data
            avg_time = np.mean(time_list)
            speedup = times["Baseline"][0] / avg_time if method != "Baseline" else 1.0
            print(f"{method}: {avg_time:.4f}s (speedup: {speedup:.2f}x)")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main() 