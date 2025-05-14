import os
import sys
import time
import torch
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Try to import necessary modules
try:
    from diffusers import StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Warning: diffusers library not properly installed, make sure you have diffusers>=0.30.0")

# Import the cache implementations
try:
    from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
except ImportError:
    print("Warning: xfuser cache modules not found, some cache methods may not work")

def parse_args():
    parser = argparse.ArgumentParser(description="FastCache Simple Test")
    parser.add_argument("--model_type", type=str, choices=["sd3", "flux"], default="sd3")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_method", type=str, choices=["None", "Fast", "Fb", "Tea"], default="Fast")
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="fastcache_test_results")
    return parser.parse_args()

def create_accelerator(module, cache_threshold=0.05, motion_threshold=0.1):
    """Create a FastCache accelerator directly without using the registration mechanism"""
    accelerator = FastCacheAccelerator(
        module, 
        cache_ratio_threshold=cache_threshold,
        motion_threshold=motion_threshold
    )
    return accelerator

def apply_direct_fastcache(model, cache_threshold=0.05, motion_threshold=0.1):
    """Apply FastCache directly to transformer blocks in the model"""
    accelerators = []
    
    # Find and apply FastCache to transformer blocks in unet
    if hasattr(model, "unet"):
        transformer = model.unet
        
        # Recursively find and apply FastCache to transformer blocks
        def apply_fastcache(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check module type
                module_type = child.__class__.__name__
                if "Transformer" in module_type or "Attention" in module_type:
                    # Create accelerator
                    accelerator = create_accelerator(
                        child, 
                        cache_threshold=cache_threshold,
                        motion_threshold=motion_threshold
                    )
                    accelerators.append((full_name, accelerator))
                    
                    # Replace original module
                    setattr(module, name, accelerator)
                    print(f"Applied FastCache to {module_type} at {full_name}")
                else:
                    # Recursively process child modules
                    apply_fastcache(child, full_name)
        
        apply_fastcache(transformer)
        print(f"Applied FastCache to {len(accelerators)} transformer blocks")
    
    return model, accelerators

def main():
    args = parse_args()
    print(f"Testing with model: {args.model} (type: {args.model_type})")
    print(f"Cache method: {args.cache_method}")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
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
        
        # Apply selected cache method
        accelerators = []
        method_name = "Baseline"
        
        if args.cache_method == "None":
            # No caching - use as baseline
            pass
        
        elif args.cache_method == "Fast":
            method_name = "FastCache"
            # Apply FastCache directly to transformer blocks
            model, accelerators = apply_direct_fastcache(
                model, 
                cache_threshold=args.cache_ratio_threshold,
                motion_threshold=args.motion_threshold
            )
            
        elif args.cache_method in ["Fb", "Tea"]:
            method_name = "FBCache" if args.cache_method == "Fb" else "TeaCache"
            # Use existing cache implementations through flux adapter
            if args.model_type == "flux" and hasattr(model, "transformer"):
                # Apply cache through the flux adapter
                apply_cache_on_transformer(
                    model.transformer,
                    rel_l1_thresh=args.cache_ratio_threshold,
                    return_hidden_states_first=False,
                    num_steps=args.num_inference_steps,
                    use_cache=args.cache_method,
                    motion_threshold=args.motion_threshold
                )
                print(f"Applied {method_name} to transformer through flux adapter")
            else:
                print(f"Warning: {method_name} not implemented for {args.model_type}")
        
        # Run inference with timing
        print(f"\nRunning inference with {method_name} ({args.num_inference_steps} steps)...")
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
        image_path = os.path.join(args.output_dir, f"{args.cache_method.lower()}_image.png")
        output_image.save(image_path)
        print(f"Image saved to {image_path}")
        
        # Save statistics to file
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
        if args.cache_method == "Fast":
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
        
        # Save stats
        stats_path = os.path.join(args.output_dir, f"{args.cache_method.lower()}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 