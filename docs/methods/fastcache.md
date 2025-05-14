# FastCache: Adaptive Spatial-Temporal Caching for Transformer Acceleration

FastCache is a novel acceleration technique for Diffusion Transformer (DiT) models that exploits computational redundancies across both spatial and temporal dimensions. It offers significant speedups while maintaining output quality by intelligently identifying and eliminating unnecessary computations.

## Overview

FastCache introduces a hidden-state-level caching and compression framework with two core components:

1. **Spatial Token Reduction Module** - Identifies and suppresses redundant tokens across spatial dimensions
2. **Transformer-Level Caching Module** - Leverages statistical tests to determine when hidden states can be reused

The approach works by analyzing the motion and variation of hidden states across denoising timesteps and spatial tokens, avoiding full transformer computations for regions of low change.

![FastCache Model Overview](https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/fastcache_overview.png)

## Key Advantages

- **Adaptive Computation**: Automatically adjusts caching behavior based on model hidden state changes
- **Zero Training Required**: Works as a drop-in acceleration without model retraining
- **Minimal Quality Impact**: Maintains high output quality by selectively applying caching
- **Memory Efficient**: Reduces peak memory usage by avoiding redundant computations
- **Compatible with Other Methods**: Can be combined with other acceleration techniques like TeaCache, First-Block-Cache, or parallel inference

## Technical Details

### Spatial Token Reduction

FastCache computes a motion-aware saliency metric by comparing hidden states between timesteps:

```
S_var = max(|X - X_prev|)
```

An adaptive threshold determines which tokens require full computation:

```
τ_adaptive = β₀ + β₁S_var + β₂t + β₃t²
```

For static tokens (below threshold), a parameter-efficient linear approximation is applied instead of full transformer computation.

### Transformer-Level Caching

For caching decisions at the transformer block level, FastCache computes a relative change metric:

```
δₜ = ‖Hₜ - Hₜ₋₁‖_F / ‖Hₜ₋₁‖_F
```

A statistical test based on chi-square distribution determines when caching can be applied safely.

## How to Use FastCache

### Basic Usage

FastCache can be easily applied to any DiT model in the xFuser framework:

```python
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from diffusers import StableDiffusion3Pipeline

# Load your diffusion model
model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")

# Create FastCache wrapper
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)

# Enable FastCache with optional parameters
fastcache_wrapper.enable_fastcache(
    cache_ratio_threshold=0.05,  # Relative change threshold for caching
    motion_threshold=0.1,        # Threshold for motion saliency
)

# Run inference with FastCache acceleration
result = fastcache_wrapper(
    prompt="a photo of an astronaut riding a horse on the moon",
    num_inference_steps=30,
)

# Get cache statistics
stats = fastcache_wrapper.get_cache_statistics()
print(stats)
```

### Benchmarking FastCache

We provide a benchmark script to compare FastCache with other acceleration methods:

```bash
python examples/fastcache_benchmark.py \
    --model_type flux \
    --model "black-forest-labs/FLUX.1-schnell" \
    --prompt "a beautiful landscape with mountains and a lake" \
    --num_inference_steps 30 \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1
```

## Performance Comparison

FastCache demonstrates significant speedups compared to baseline DiT inference:

| Model | Baseline | FastCache | TeaCache | First-Block-Cache |
|-------|----------|-----------|----------|------------------|
| SD3-Medium | 12.4s | 7.3s (1.7x) | N/A | N/A |
| Flux.1 | 9.8s | 6.2s (1.6x) | 7.1s (1.4x) | 7.5s (1.3x) |
| PixArt Sigma | 10.6s | 6.7s (1.6x) | N/A | N/A |

FastCache generally outperforms other single-GPU acceleration methods while maintaining high output quality.

## Integration with Parallel Inference

FastCache is designed to work alongside xDiT's parallel inference methods for even greater speedups. To use FastCache with parallel inference:

```python
# Enable FastCache in your pipeline
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
fastcache_wrapper.enable_fastcache()

# Apply parallel inference (USP, PipeFusion, etc.)
paralleler = xDiTParallel(fastcache_wrapper, engine_config, input_config)

# Run inference with both FastCache and parallel acceleration
result = paralleler(...)
```

## Limitations and Future Work

- Currently optimized for image generation DiT models; video support coming soon
- Performance gains vary based on model architecture and complexity of the generated content
- Future work will focus on improving the spatial token selection mechanism and exploring learned linear projections for better approximation of static tokens

## Citation

If you use FastCache in your research or applications, please cite:

```bibtex
@article{fang2024fastcache,
  title={FastCache: Adaptive Spatial-Temporal Caching for Diffusion Transformer Acceleration},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
``` 