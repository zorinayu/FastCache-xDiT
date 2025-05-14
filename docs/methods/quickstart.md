# FastCache-xDiT: 快速入门指南

本指南将帮助您开始使用 FastCache-xDiT，一个用于加速扩散变换器(DiT)模型的即插即用框架。

## 安装步骤

### 要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU加速)

### 安装方法

```bash
# 基础安装
pip install xfuser

# 包含diffusers和flash-attention的完整安装
pip install "xfuser[diffusers,flash-attn]"

# 从源码安装
git clone https://github.com/xdit-project/xDiT.git
cd xDiT
pip install -e ".[diffusers,flash-attn]"
```

## 运行 FastCache

### 基本用法

FastCache-xDiT 可以通过Python API或命令行使用。

#### Python API 示例

```python
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from diffusers import StableDiffusion3Pipeline

# 加载模型
model = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")

# 创建FastCache包装器
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)

# 启用FastCache并配置参数
fastcache_wrapper.enable_fastcache(
    cache_ratio_threshold=0.05,  # 缓存相对变化阈值
    motion_threshold=0.1,        # 运动显著性阈值
)

# 运行推理
result = fastcache_wrapper(
    prompt="a photo of an astronaut riding a horse on the moon",
    num_inference_steps=30,
)

# 查看缓存统计信息
stats = fastcache_wrapper.get_cache_statistics()
print(f"Cache hit ratio: {sum(s['cache_hit_ratio'] for s in stats.values())/len(stats):.2%}")

# 保存结果
result.images[0].save("fastcache_result.png")
```

### 命令行用法

我们提供了两种命令行运行FastCache的方式：

#### 方式一：简化版基准测试脚本（推荐新用户使用）

简化版脚本提供了更好的错误处理和更详细的运行日志：

```bash
# 在Stable Diffusion 3上运行
./examples/run_fastcache_simple.sh

# 在Flux模型上运行
./examples/run_fastcache_simple.sh flux
```

如果你在运行标准脚本时遇到导入错误或其他问题，建议使用这个简化版脚本。

#### 方式二：标准基准测试脚本

用于全面比较FastCache与其他缓存方法的性能：

```bash
# 在Stable Diffusion 3上运行
./examples/run_fastcache_benchmark.sh

# 在Flux模型上运行
./examples/run_fastcache_benchmark.sh flux
```

#### 直接使用Python脚本

如果需要更详细的参数控制，可以直接使用Python脚本：

```bash
# 在Stable Diffusion 3上运行
python examples/fastcache_benchmark_simple.py \
    --model_type sd3 \
    --model "stabilityai/stable-diffusion-3-medium-diffusers" \
    --prompt "a photo of an astronaut riding a horse on the moon" \
    --num_inference_steps 30 \
    --cache_ratio_threshold 0.05 \
    --motion_threshold 0.1 \
    --output_dir "sd3_results"

# 在Flux上运行
python examples/fastcache_benchmark_simple.py \
    --model_type flux \
    --model "black-forest-labs/FLUX.1-schnell" \
    --prompt "a beautiful landscape with mountains" \
    --num_inference_steps 30 \
    --output_dir "flux_results"
```

### 重要参数说明

| 参数 | 描述 | 推荐值范围 |
|------|------|------------|
| `cache_ratio_threshold` | 控制缓存决策的相对变化阈值。较低的值意味着更严格的缓存条件，但可能带来更好的质量 | 0.01 - 0.1 |
| `motion_threshold` | 控制空间token选择的运动显著性阈值。较低的值将处理更多token，较高的值将跳过更多token | 0.05 - 0.2 |
| `num_inference_steps` | 扩散步骤数。步骤越多，FastCache的加速效果通常越明显 | 20 - 50 |

## 常见问题解答

### FastCache如何影响图像质量？

FastCache设计为在保持输出质量的同时加速推理。在默认参数下，质量变化通常是不可察觉的。如果您发现质量下降，可以尝试降低`cache_ratio_threshold`和`motion_threshold`的值。

### 如何测量FastCache的加速效果？

使用`examples/fastcache_benchmark.py`运行基准测试，它会比较有无FastCache的运行时间。结果会保存在输出目录中，包括时间比较图表和各种缓存方法产生的图像。

### 如何调整FastCache参数？

- 增加`cache_ratio_threshold`会提高加速效果，但可能影响质量
- 增加`motion_threshold`会跳过更多静态token，提高速度
- 这些参数可以根据您的具体模型和用例进行调整

### 在哪些模型上FastCache效果最好？

FastCache在具有较多冗余计算的DiT模型上效果最佳，特别是：
- Stable Diffusion 3
- FLUX.1
- PixArt-Sigma
- 其他基于DiT架构的模型

## 与并行方法结合

FastCache可以与xDiT的并行方法结合使用，进一步提升性能：

```python
# 启用FastCache
fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
fastcache_wrapper.enable_fastcache()

# 创建并行配置
from xfuser import xFuserArgs
from xfuser.parallel import xDiTParallel
from xfuser.config import FlexibleArgumentParser

# 创建参数
parser = FlexibleArgumentParser()
args = xFuserArgs.add_cli_args(parser).parse_args([
    "--ulysses_degree", "2",
    "--use_cfg_parallel"  # 使用CFG并行(2个GPU)
])
engine_args = xFuserArgs.from_cli_args(args)
engine_config, input_config = engine_args.create_config()

# 创建并行执行器
paralleler = xDiTParallel(fastcache_wrapper, engine_config, input_config)

# 运行加速推理
result = paralleler(
    prompt="a beautiful landscape with mountains",
    num_inference_steps=30
)
```

## 进一步阅读

- [FastCache技术细节](./fastcache.md)
- [支持的DiT模型](../README.md#support-dits)
- [xDiT并行方法](../README.md#parallel) 