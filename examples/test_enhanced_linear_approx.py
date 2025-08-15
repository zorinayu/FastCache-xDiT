#!/usr/bin/env python3
"""
Enhanced Linear Approximation Algorithm Demo

This script demonstrates the enhanced FastCache Linear Approximation Algorithm
that combines both block-level and token-level linear approximations.

Algorithm: FastCache Linear Approximation Framework
- Block Level: Statistical change detection with chi-square test
- Token Level: Motion-aware token masking with learnable linear projections
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path
import json

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from diffusers import FluxPipeline, PixArtSigmaPipeline
    from PIL import Image
except ImportError:
    print("Please install diffusers>=0.30.0 and Pillow")
    sys.exit(1)

def test_enhanced_linear_approximation(
    model_type="flux",
    model_name="black-forest-labs/FLUX.1-schnell",
    prompt="a serene landscape with mountains and a lake",
    num_inference_steps=30,
    cache_ratio_threshold=0.05,
    motion_threshold=0.1,
    enable_enhanced_linear_approx=True,
    significance_level=0.05,
    output_dir="enhanced_linear_approx_results"
):
    """
    Test the enhanced Linear Approximation Algorithm
    
    Args:
        model_type: Type of model ("flux", "pixart")
        model_name: Model name or path
        prompt: Text prompt for generation
        num_inference_steps: Number of inference steps
        cache_ratio_threshold: Threshold for block-level caching
        motion_threshold: Threshold for token-level motion detection
        enable_enhanced_linear_approx: Whether to enable enhanced algorithm
        significance_level: Statistical significance level for chi-square test
        output_dir: Output directory for results
    """
    
    print(f"Testing Enhanced Linear Approximation Algorithm")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Enhanced Linear Approximation: {enable_enhanced_linear_approx}")
    print(f"Significance Level: {significance_level}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_type == "flux":
        pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_type == "pixart":
        pipeline = PixArtSigmaPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    pipeline = pipeline.to("cuda")
    
    # Test baseline (no caching)
    print("Running baseline (no caching)...")
    start_time = time.time()
    
    with torch.no_grad():
        baseline_result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
    
    baseline_time = time.time() - start_time
    print(f"Baseline time: {baseline_time:.2f}s")
    
    # Save baseline image
    baseline_image = baseline_result.images[0]
    baseline_image.save(os.path.join(output_dir, "baseline.png"))
    
    # Test enhanced FastCache
    print("Running enhanced FastCache...")
    start_time = time.time()
    
    # Apply enhanced FastCache
    from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer
    
    # Find transformer in the pipeline
    if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "transformer"):
        transformer = pipeline.unet.transformer
    else:
        print("Warning: Could not find transformer in pipeline")
        return
    
    # Apply enhanced FastCache
    apply_cache_on_transformer(
        transformer,
        use_cache="Fast",
        rel_l1_thresh=cache_ratio_threshold,
        motion_threshold=motion_threshold,
        enable_enhanced_linear_approx=enable_enhanced_linear_approx,
        significance_level=significance_level,
        return_hidden_states_first=False,
        num_steps=num_inference_steps
    )
    
    with torch.no_grad():
        enhanced_result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
    
    enhanced_time = time.time() - start_time
    print(f"Enhanced FastCache time: {enhanced_time:.2f}s")
    
    # Save enhanced image
    enhanced_image = enhanced_result.images[0]
    enhanced_image.save(os.path.join(output_dir, "enhanced_fastcache.png"))
    
    # Calculate speedup
    speedup = baseline_time / enhanced_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Save comparison image
    comparison_image = Image.new('RGB', (baseline_image.width * 2, baseline_image.height))
    comparison_image.paste(baseline_image, (0, 0))
    comparison_image.paste(enhanced_image, (baseline_image.width, 0))
    comparison_image.save(os.path.join(output_dir, "comparison.png"))
    
    # Save results
    results = {
        "model_type": model_type,
        "model_name": model_name,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "enable_enhanced_linear_approx": enable_enhanced_linear_approx,
        "significance_level": significance_level,
        "cache_ratio_threshold": cache_ratio_threshold,
        "motion_threshold": motion_threshold,
        "baseline_time": baseline_time,
        "enhanced_time": enhanced_time,
        "speedup": speedup
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Speedup achieved: {speedup:.2f}x")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test Enhanced Linear Approximation Algorithm")
    parser.add_argument("--model_type", type=str, default="flux", choices=["flux", "pixart"],
                       help="Model type")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="a serene landscape with mountains and a lake",
                       help="Text prompt")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05,
                       help="Cache ratio threshold")
    parser.add_argument("--motion_threshold", type=float, default=0.1,
                       help="Motion threshold")
    parser.add_argument("--enable_enhanced", action="store_true",
                       help="Enable enhanced linear approximation")
    parser.add_argument("--significance_level", type=float, default=0.05,
                       help="Statistical significance level")
    parser.add_argument("--output_dir", type=str, default="enhanced_linear_approx_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    test_enhanced_linear_approximation(
        model_type=args.model_type,
        model_name=args.model,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        cache_ratio_threshold=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
        enable_enhanced_linear_approx=args.enable_enhanced,
        significance_level=args.significance_level,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
