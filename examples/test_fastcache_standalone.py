#!/usr/bin/env python3

import os
import sys
import time
import torch
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from diffusers import PixArtSigmaPipeline
except ImportError:
    print("diffusers not available. Please install diffusers>=0.30.0")
    sys.exit(1)

# Import our standalone FastCache wrapper
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper

def main():
    print("Testing standalone FastCache wrapper...")
    
    # Load a small PixArt model
    model_name = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    prompt = "a photo of an astronaut riding a horse on the moon"
    
    print(f"Loading model: {model_name}")
    try:
        pipeline = PixArtSigmaPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create FastCache wrapper
    print("Creating FastCache wrapper...")
    try:
        fastcache_wrapper = xFuserFastCachePipelineWrapper(pipeline)
        print("FastCache wrapper created successfully")
        
        # Enable FastCache
        print("Enabling FastCache...")
        fastcache_wrapper.enable_fastcache(
            cache_ratio_threshold=0.05,
            motion_threshold=0.1,
            significance_level=0.05
        )
        print("FastCache enabled successfully")
        
    except Exception as e:
        print(f"Error creating FastCache wrapper: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run a quick test
    print("Running FastCache inference test...")
    try:
        generator = torch.Generator(device="cuda").manual_seed(42)
        
        start_time = time.time()
        result = fastcache_wrapper(
            prompt=prompt,
            num_inference_steps=5,  # Short test
            height=256,
            width=256,
            generator=generator,
        )
        end_time = time.time()
        
        print(f"FastCache inference completed in {end_time - start_time:.2f} seconds")
        
        # Get cache statistics
        stats = fastcache_wrapper.get_cache_statistics()
        print(f"Cache statistics: {stats}")
        
        # Save result
        if hasattr(result, 'images') and len(result.images) > 0:
            result.images[0].save("fastcache_test_result.png")
            print("Test image saved to fastcache_test_result.png")
        
        print("FastCache test completed successfully!")
        
    except Exception as e:
        print(f"Error during FastCache inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 