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

# Import the cache modules directly
try:
    from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator
    from xfuser.model_executor.cache.utils import FBCachedTransformerBlocks, TeaCachedTransformerBlocks
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
except ImportError:
    print("Warning: xfuser cache modules not found")

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

def apply_fastcache(model, cache_threshold=0.05, motion_threshold=0.1):
    """Apply FastCache to transformer blocks in the model"""
    accelerators = []
    
    # Find and apply FastCache to transformer blocks
    target_module = None
    if hasattr(model, "unet"):
        target_module = model.unet
    elif hasattr(model, "transformer"):
        target_module = model.transformer
    
    if target_module is not None:
        # Recursively find and apply FastCache to transformer blocks
        def apply_cache(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check module type
                module_type = child.__class__.__name__
                if "Transformer" in module_type or "Attention" in module_type:
                    # Create accelerator
                    accelerator = FastCacheAccelerator(
                        child, 
                        cache_ratio_threshold=cache_threshold,
                        motion_threshold=motion_threshold
                    )
                    accelerators.append((full_name, accelerator))
                    
                    # Replace original module
                    setattr(module, name, accelerator)
                    print(f"Applied FastCache to {module_type} at {full_name}")
                else:
                    # Recursively process child modules
                    apply_cache(child, full_name)
        
        apply_cache(target_module)
        print(f"Applied FastCache to {len(accelerators)} transformer blocks")
    else:
        print("WARNING: Could not find suitable transformer module in the model")
    
    return model, accelerators

def get_model_for_method(args, method, original_model=None):
    """Get a model instance with the specified cache method applied"""
    if original_model is not None and method == "None":
        # Return the original model for baseline
        return original_model, None
    
    # Load a fresh model instance
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
    
    # Apply cache method
    accelerators = None
    
    if method == "None":
        # No caching - use as baseline
        pass
    
    elif method == "Fast":
        # Apply FastCache directly
        model, accelerators = apply_fastcache(
            model, 
            cache_threshold=args.cache_ratio_threshold,
            motion_threshold=args.motion_threshold
        )
    
    elif method in ["Fb", "Tea"]:
        # Apply FB/Tea cache using flux adapter
        if hasattr(model, "transformer"):
            # Try to use the flux adapter
            apply_cache_on_transformer(
                model.transformer,
                rel_l1_thresh=args.cache_ratio_threshold,
                return_hidden_states_first=False,
                num_steps=args.num_inference_steps,
                use_cache=method,
                motion_threshold=args.motion_threshold
            )
            print(f"Applied {method}Cache to transformer through flux adapter")
        else:
            print(f"WARNING: Could not apply {method}Cache to model (no transformer found)")
    
    return model, accelerators

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
        model, accelerators = get_model_for_method(args, method, baseline_model)
        
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
        
        # Add cache-specific stats for FastCache
        if method == "Fast" and accelerators:
            cache_stats = {}
            total_hits = 0
            total_steps = 0
            
            for name, acc in accelerators:
                hits = acc.cache_hits
                steps = acc.total_steps
                total_hits += hits
                total_steps += steps
                if steps > 0:
                    cache_stats[name] = {
                        "hits": int(hits),
                        "steps": int(steps),
                        "hit_ratio": float(hits/steps)
                    }
            
            if total_steps > 0:
                cache_stats["overall"] = {
                    "hits": int(total_hits),
                    "steps": int(total_steps),
                    "hit_ratio": float(total_hits/total_steps)
                }
            
            stats["cache_stats"] = cache_stats
            stats["cache_threshold"] = args.cache_ratio_threshold
            stats["motion_threshold"] = args.motion_threshold
            
            if total_steps > 0:
                print(f"\nCache hit statistics:")
                print(f"Overall: {total_hits}/{total_steps} hits ({total_hits/total_steps:.2%})")
        
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