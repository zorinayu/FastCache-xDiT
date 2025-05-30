# FastCache: Linear Approximation-Based Acceleration for Diffusion Transformers

FastCache is a novel acceleration technique for Diffusion Transformer (DiT) models that leverages **learnable linear approximations** to replace expensive transformer computations. By intelligently identifying redundant computations and substituting them with efficient linear projections, FastCache achieves significant speedups while maintaining output quality.

## Core Innovation: Linear Approximation Framework

The fundamental insight behind FastCache is that many computations in diffusion transformers exhibit high redundancy across spatial tokens and temporal steps. Instead of performing full transformer operations on redundant data, FastCache introduces **learnable linear approximations** that can efficiently approximate these computations with minimal quality loss.

FastCache operates through two complementary linear approximation modules:

1. **Spatial Linear Approximation** - Replaces transformer processing of static tokens with learnable linear projections
2. **Temporal Linear Approximation** - Substitutes entire transformer blocks with block-specific linear transformations when hidden states show minimal change


<picture>
  <img alt="FastCache Architecture" src="../../assets/architecture.png" width="80%">
</picture>

<picture>
  <img alt="FastCache Interpretability" src="../../assets/overview.png" width="80%">
</picture>

<picture>
  <img alt="FastCache Model Design" src="../../assets/DiTLevelCache-interp.png" width="80%">
</picture>

## Technical Framework: Dual-Level Linear Approximation

### 1. Motion-Aware Token Masking

FastCache first identifies which input tokens are likely to produce significant activation changes and which are static. This decision is made before any transformer block computation.

Let $X_t \in R^{N \times D}$ be the input tokens at timestep $t$, and $X_{t-1}$ be those from the previous timestep. We compute a motion-based saliency score for each token:

$S_t^{(i)} = \| X_t^{(i)} - X_{t-1}^{(i)} \|_2^2$

Tokens are partitioned as:

- **Dynamic Tokens** $M_t$: High motion saliency → pass through full transformer stack
- **Static Tokens** $S_t$: Low motion saliency → replaced with linear approximations


### 2. Transformer Block Linear Approximation

At the transformer block level, FastCache determines when entire blocks can be replaced with linear approximations.

**Statistical Change Detection:**
```
δ_t,l = ||H_t,l-1 - H_{t-1,l-1||_F / ||H_{t-1,l-1}||_F  # Relative change metric
```

**Chi-Square Statistical Test:**
Under the assumption that hidden state changes follow a scaled chi-square distribution:
```
(N×D) × δ_t,l² ~ χ²_{N×D}
```

**Block-Level Linear Approximation:**
When the statistical test indicates minimal change (δ_t,l² ≤ threshold), the entire transformer block is replaced with a block-specific linear approximation:

```
H_t,l = W_block_l × H_t,l-1 + b_block_l
```

Where `W_block_l` and `b_block_l` are learnable parameters specific to transformer block `l`. This replaces the expensive multi-head attention and feed-forward computations with a simple linear transformation.

## Linear Approximation Algorithm

Block Level
```
Algorithm: FastCache Linear Approximation Framework
Input: Hidden state H_t, previous H_{t-1}, learnable parameters {W, b}
Output: Approximated hidden state H_t^L

    // Transformer Block Linear Approximation  
   For l = 1 to L:
     δ_{t,l} ← ||H_{t,l-1} - H_{t-1,l-1}||_F / ||H_{t-1,l-1}||_F
     
     If δ_{t,l}² ≤ χ²_{ND,1-α}/ND:
       // Apply linear approximation
       H_{t,l} ← W_block_l × H_{t,l-1} + b_block_l
     Else:
       // Full transformer computation
       H_{t,l} ← TransformerBlock_l(H_{t,l-1})

3. Return H_t^L
```

Token-Level
```
Algorithm: FastCache Linear Approximation with Masking (Corrected)
Input:
    Token embedding X_t, previous X_{t-1}
    Previous hidden states {H_{t-1,l}} for l = 1..L
    Learnable parameters {W_static, b_static, W_block_l, b_block_l}
Output:
    Final hidden state H_{t,L}

1. // === Token-Level Saliency Mask ===
   Compute saliency: S_t = ||X_t - X_{t-1}||²
   Token mask:
       M_token = {i | S_t[i] > τ_motion}     # motion tokens
       S_token = {i | S_t[i] ≤ τ_motion}     # static tokens

   # Precompute output for static tokens (final)
   H_static = W_static × X_t[S_token] + b_static

2. // === Transformer Stack on Motion Tokens Only ===
   Initialize H_motion ← embed(X_t[M_token])

   For l = 1 to L:
       Compute relative change for block:
           δ_{t,l} = ||H_{t,l-1} - H_{t-1,l-1}||_F / ||H_{t-1,l-1}||_F

       If δ_{t,l}² ≤ χ²_{ND,1-α}/ND:
           # Only static tokens take linear approximation through this block
           H_static_l = W_block_l × H_static + b_block_l
       Else:
           # Motion tokens enter full transformer block
           H_motion = TransformerBlock_l(H_motion)

3. // === Final Composition ===
   Initialize H_t^L ← zeros(N, D)
   H_t^L[S_token] ← H_static_l
   H_t^L[M_token] ← H_motion

4. Return H_t^L
```

## Key Advantages of Linear Approximation Approach

- **Computational Efficiency**: Linear operations (O(d)) replace quadratic attention computations (O(d²))
- **Memory Efficiency**: Eliminates intermediate activations for approximated computations
- **Adaptive Precision**: Automatically balances speed vs. quality based on content complexity
- **Learnable Parameters**: Linear projections can be fine-tuned to minimize approximation error
- **Plug-and-Play**: Works with any transformer architecture without structural modifications

## Performance Analysis: Linear Approximation Impact

The linear approximation framework provides significant computational savings:

**Spatial Token Reduction:**
- Baseline: All tokens processed through full transformer stack
- FastCache: Only motion tokens (typically 20-40%) require full processing
- Speedup: 2.5-5x reduction in spatial computation

**Transformer Block Approximation:**
- Baseline: All blocks computed for every timestep
- FastCache: 60-80% of blocks replaced with linear approximations
- Speedup: 3-5x reduction in transformer computation

**Combined Speedup:**
| Model | Baseline | FastCache | Linear Approx. Ratio |
|-------|----------|-----------|---------------------|
| SD3-Medium | 12.4s | 7.3s (1.7x) | 75% blocks, 65% tokens |
| Flux.1 | 9.8s | 6.2s (1.6x) | 70% blocks, 60% tokens |
| PixArt Sigma | 10.6s | 6.7s (1.6x) | 72% blocks, 62% tokens |

## Usage: Implementing Linear Approximation

### Basic Usage

```python
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from diffusers import StableDiffusion3Pipeline

# Load your diffusion model
model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")

# Create FastCache wrapper with linear approximation
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)

# Enable FastCache linear approximation
fastcache_wrapper.enable_fastcache(
    cache_ratio_threshold=0.05,  # Threshold for block-level linear approximation
    motion_threshold=0.1,        # Threshold for spatial linear approximation
    significance_level=0.05,     # Statistical confidence for linear approximation
)

# Run inference with linear approximation acceleration
result = fastcache_wrapper(
    prompt="a photo of an astronaut riding a horse on the moon",
    num_inference_steps=30,
)

# Analyze linear approximation statistics
stats = fastcache_wrapper.get_cache_statistics()
print(f"Spatial linear approximation ratio: {stats['spatial_approx_ratio']:.2%}")
print(f"Block linear approximation ratio: {stats['block_approx_ratio']:.2%}")
```

### Advanced Configuration

```python
# Fine-tune linear approximation parameters
fastcache_wrapper.enable_fastcache(
    cache_ratio_threshold=0.03,    # More aggressive block approximation
    motion_threshold=0.15,         # More conservative spatial approximation
    adaptive_beta=[0.1, 0.2, 0.05, 0.01],  # Custom adaptive threshold parameters
    enable_spatial_approx=True,    # Enable spatial linear approximation
    enable_block_approx=True,      # Enable block linear approximation
)
```

## Benchmarking Linear Approximation Performance

```bash
# Compare linear approximation methods
python examples/fastcache_benchmark.py \
    --model_type flux \
    --model "black-forest-labs/FLUX.1-schnell" \
    --prompt "a beautiful landscape with mountains and a lake" \
    --num_inference_steps 30 \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1 \
    --analyze_approximation_quality
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

The linear approximation framework is fully compatible with parallel inference:

```python
# Combine linear approximation with parallel inference
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
fastcache_wrapper.enable_fastcache()

# Apply parallel inference with linear approximation benefits
paralleler = xDiTParallel(fastcache_wrapper, engine_config, input_config)
result = paralleler(...)  # Benefits from both linear approximation and parallelization
```

## Future Directions: Advanced Linear Approximation

- **Learned Linear Projections**: Training specialized linear approximations for different model architectures
- **Hierarchical Approximation**: Multi-level linear approximations for even greater efficiency
- **Dynamic Approximation**: Real-time adaptation of linear approximation parameters based on content complexity
- **Cross-Modal Approximation**: Extending linear approximation techniques to video and 3D generation models

## Citation

If you use FastCache's linear approximation framework in your research, please cite:

```bibtex
@inproceedings{liu2025fastcache,
  title={FastCache: Cache What Matters, Skip What Doesn't.},
  author={Liu, Dong and Zhang, Jiayi and Li, Yifan and Yu, Yanxuan and Lengerich, Ben and Wu, Ying Nian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2025}
}
@article{liu2025fastcache,
  title={FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation},
  author={Liu, Dong and Zhang, Jiayi and Li, Yifan and Yu, Yanxuan and Lengerich, Ben and Wu, Ying Nian},
  journal={arXiv preprint arXiv:2505.20353},
  year={2025}
}
```