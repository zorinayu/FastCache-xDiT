<div align="center">
  <!-- <h1>KTransformers</h1> -->
  <p align="center">

  <picture>
    <img alt="FastCache-xDiT" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/XDiTlogo.png" width="50%">

  </p>
  <h3>FastCache-xDiT: A Plug-and-Play Acceleration Method for Diffusion Transformers</h3>
  <a href="#cite-us">📝 Papers</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported DiTs</a> | <a href="#fastcache-overview">⚡ FastCache</a> | <a href="https://github.com/xdit-project/xDiT/discussions">📈 Discussion </a> | <a href="https://medium.com/@xditproject">📝 Blogs</a></strong>
  <p></p>

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/YEWzWfCF9S)](https://discord.gg/YEWzWfCF9S)

</div>

<h2 id="agenda">Table of Contents</h2>

- [🔥 FastCache-xDiT](#fastcache-overview)
- [🚀 QuickStart](#QuickStart)
- [🎯 Supported DiTs](#support-dits)
- [📊 Performance Comparison](#performance-comparison)
- [🧠 Technical Details](#technical-details) 
- [📈 xDiT's Parallel Methods](#parallel)
- [🔍 Single GPU Acceleration](#1gpuacc)
- [📚 Develop Guide](#dev-guide)
- [📝 Cite Us](#cite-us)

<h2 id="fastcache-overview">🔥 FastCache-xDiT</h2>

FastCache-xDiT is a novel **plug-and-play acceleration method** for Diffusion Transformers (DiTs) that exploits computational redundancies across both spatial and temporal dimensions. With **zero training required** and **minimal quality impact**, FastCache can deliver significant speedups (up to 1.7x) on modern DiT models while being fully compatible with existing parallel inference methods.

### Key Features

- **Plug-and-Play**: Drop-in acceleration with no model modifications required
- **Adaptive Computation**: Dynamically adjusts caching behavior based on model hidden states
- **Spatial-Temporal Awareness**: Intelligently identifies redundant computations in both dimensions
- **Memory Efficient**: Reduces peak memory usage by avoiding unnecessary computations
- **Compatible with Parallel Methods**: Can be combined with USP, PipeFusion, and other xDiT parallel techniques

FastCache introduces a hidden-state-level caching and compression framework with two core components:

1. **Spatial Token Reduction Module** - Adaptively identifies and processes only tokens with significant changes
2. **Transformer-Level Caching Module** - Uses statistical tests to determine when entire transformer blocks can be skipped

<picture>
  <img alt="FastCache Design" src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/fastcache_overview.png" width="80%">
</picture>

<h2 id="performance-comparison">📊 Performance Comparison</h2>

FastCache delivers significant speedups across popular DiT models:

| Model | Baseline | FastCache | TeaCache | First-Block-Cache |
|-------|----------|-----------|----------|------------------|
| SD3-Medium | 12.4s | 7.3s (1.7x) | N/A | N/A |
| Flux.1 | 9.8s | 6.2s (1.6x) | 7.1s (1.4x) | 7.5s (1.3x) |
| PixArt Sigma | 10.6s | 6.7s (1.6x) | N/A | N/A |

<h2 id="technical-details">🧠 Technical Details</h2>

FastCache-xDiT operates on two levels:

### Spatial Token Reduction

FastCache computes a motion-aware saliency metric by comparing hidden states between timesteps:

```
S_var = max(|X - X_prev|)
```

An adaptive threshold determines which tokens require full computation:

```
τ_adaptive = β₀ + β₁S_var + β₂t + β₃t²
```

### Transformer-Level Caching

For caching decisions at the transformer block level, FastCache computes a relative change metric:

```
δₜ = ‖Hₜ - Hₜ₋₁‖_F / ‖Hₜ₋₁‖_F
```

A statistical test based on chi-square distribution determines when caching can be applied safely:

```
(ND)δₜ² ≤ χ²ₙₘ₂,₁₋α
```

For more details, see our [FastCache documentation](./docs/methods/fastcache.md).

<h2 id="QuickStart">🚀 QuickStart</h2>

### 1. Install xFuser

```bash
pip install xfuser  # Basic installation
pip install "xfuser[diffusers,flash-attn]"  # With both diffusers and flash attention
```

### 2. Using FastCache Acceleration

#### Python API

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

#### Command Line Usage

Run FastCache with Stable Diffusion 3:

```bash
# Basic usage
python examples/run_fastcache_test.py \
    --model_type sd3 \
    --model "stabilityai/stable-diffusion-3-medium-diffusers" \
    --prompt "a photo of an astronaut riding a horse on the moon" \
    --num_inference_steps 30 \
    --cache_method "Fast" \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1

# Using the convenience benchmark script to compare different cache methods
./examples/run_fastcache_benchmark.sh sd3
```

Run FastCache with Flux model:

```bash
# Basic usage
python examples/run_fastcache_test.py \
    --model_type flux \
    --model "black-forest-labs/FLUX.1-schnell" \
    --prompt "a serene landscape with mountains and a lake" \
    --num_inference_steps 30 \
    --cache_method "Fast" \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1

# Using the convenience benchmark script
./examples/run_fastcache_benchmark.sh flux
```

#### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model type (`sd3` or `flux`) | `sd3` |
| `--model` | Model path or name | `stabilityai/stable-diffusion-3-medium-diffusers` |
| `--prompt` | Text prompt for image generation | `a photo of an astronaut riding a horse on the moon` |
| `--num_inference_steps` | Number of inference steps | `30` |
| `--cache_method` | Cache method (`None`, `Fast`, `Fb`, `Tea`) | `Fast` |
| `--seed` | Random seed | `42` |
| `--height` | Image height | `768` |
| `--width` | Image width | `768` |
| `--cache_ratio_threshold` | Cache ratio threshold | `0.05` |
| `--motion_threshold` | FastCache motion threshold | `0.1` |
| `--output_dir` | Output directory for results | `fastcache_test_results` |

### 3. Benchmark FastCache

Compare FastCache with other acceleration methods:

```bash
# Run on Stable Diffusion 3
./examples/run_fastcache_benchmark.sh sd3

# Run on Flux model
./examples/run_fastcache_benchmark.sh flux
```

The benchmark will:
- Run baseline model without acceleration (cache_method="None")
- Run with FastCache acceleration (cache_method="Fast")
- Run with First-Block-Cache acceleration (cache_method="Fb")
- Run with TeaCache acceleration (cache_method="Tea")
- Generate comparison images for quality assessment
- Create performance statistics and cache hit ratio charts
- Generate a comprehensive HTML report with all comparisons

All results will be saved to the `fastcache_benchmark_results` directory, making it easy to compare the different caching methods in terms of both performance and output quality.

### 4. Combining with Parallel Methods

FastCache can be combined with xDiT's parallel methods for even greater speedups:

```python
# Enable FastCache in your pipeline
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
fastcache_wrapper.enable_fastcache()

# Apply parallel inference (USP, PipeFusion, etc.)
engine_config, input_config = engine_args.create_config()
paralleler = xDiTParallel(fastcache_wrapper, engine_config, input_config)

# Run inference with both FastCache and parallel acceleration
result = paralleler(prompt="your prompt", num_inference_steps=30)
```

<h2 id="support-dits">🎯 Supported DiTs</h2>

<div align="center">

| Model Name | FastCache | CFG | SP | PipeFusion | Performance Report Link |
| --- | --- | --- | --- | --- | --- |
| [🟠 Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | ✅ | ✔️ | ✔️ | ✔️ | [Report](./docs/performance/sd3.md) |
| [🟠 Flux](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | ✅ | NA | ✔️ | ✔️ | [Report](./docs/performance/flux.md) |
| [🔴 PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | ✅ | ✔️ | ✔️ | ✔️ | [Report](./docs/performance/pixart_alpha_legacy.md) |
| [🎬 HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) | ✅ | ✔️ | ✔️ | ✔️ | [Report](./docs/performance/hunyuandit.md) |
| [🎬 StepVideo](https://huggingface.co/stepfun-ai/stepvideo-t2v) | ✅ | NA | ✔️ | ❎ | [Report](./docs/performance/stepvideo.md) |
| [🟢 PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha) | ✅ | ✔️ | ✔️ | ✔️ | [Report](./docs/performance/pixart_alpha_legacy.md) |
| [🎬 ConsisID-Preview](https://github.com/PKU-YuanGroup/ConsisID) | ✅ | ✔️ | ✔️ | ❎ | [Report](./docs/performance/consisid.md) |
| [🎬 CogVideoX1.5](https://huggingface.co/THUDM/CogVideoX1.5-5B) | ✅ | ✔️ | ✔️ | ❎ | [Report](./docs/performance/cogvideo.md) |

</div>

<h2 id="parallel">📈 xDiT's Parallel Methods</h2>

FastCache-xDiT is fully compatible with the parallel acceleration methods provided by xDiT:

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/xdit_method.png" alt="xdit methods">
</div>

<h4 id="PipeFusion">PipeFusion</h4>

[PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models](./docs/methods/pipefusion.md)

<h4 id="USP">Unified Sequence Parallelism (USP)</h4>

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](./docs/methods/usp.md)

<h4 id="hybrid_parallel">Hybrid Parallel</h4>

[Hybrid Parallelism](./docs/methods/hybrid.md)

<h2 id="1gpuacc">🔍 Single GPU Acceleration</h2>

<h4 id="cache_acceleration">Cache Acceleration</h4>

xDiT offers multiple cache-based acceleration methods:

1. **FastCache**: Our adaptive spatial-temporal caching method that uses motion-aware token reduction and statistical caching to exploit computational redundancies. [Read more about FastCache](./docs/methods/fastcache.md).

2. **TeaCache**: Memory-friendly caching and generation to exploit redundancies between adjacent denoising steps.

3. **First-Block-Cache**: Caches the output of early transformer blocks across timesteps.

4. **DiTFastAttn**: Reduces attention computation by leveraging redundancies between different steps of the Diffusion Model.

[DiTFastAttn: Attention Compression for Diffusion Transformer Models](./docs/methods/ditfastattn.md)

<h4 id="compilation">Compilation Acceleration</h4>

We utilize two compilation acceleration techniques, [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [onediff](https://github.com/siliconflow/onediff), to enhance runtime speed on GPUs. These compilation accelerations can be used alongside FastCache for even greater performance.

<h2 id="dev-guide">📚 Develop Guide</h2>

We provide a step-by-step guide for adding new models, please refer to the following tutorial.

[Apply xDiT to new models](./docs/developer/adding_models/readme.md)

A high-level design of xDiT framework is provided below, which may help you understand the xDiT framework.

[The implement and design of xdit framework](./docs/developer/The_implement_design_of_xdit_framework.md)

<h2 id="cite-us">📝 Cite Us</h2>

If you use FastCache-xDiT in your research or applications, please cite:

```bibtex
@inproceedings{liu2025fastcache,
  title={FastCache: Adaptive Spatial-Temporal Caching for Diffusion Transformer Acceleration},
  author={Liu, Dong and Zhang, Jiayi and Li, Yifan and Yu, Yanxuan and Lengerich, Ben and Wu, Ying Nian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={},
  year={2025}
}
```
