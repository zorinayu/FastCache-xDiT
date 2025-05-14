# 缓存方法对比工具

这个目录包含了用于对比不同缓存加速方法性能的工具和脚本。

## 可用的缓存方法

目前支持以下几种缓存方法：

1. **FastCache** - 自适应的空间-时间缓存方法，利用运动感知的token归约和统计缓存来利用计算冗余
2. **TeaCache** - 记忆友好的缓存和生成方法，利用相邻降噪步骤之间的冗余
3. **FBCache (First-Block-Cache)** - 缓存早期transformer块在时间步之间的输出
4. **DiTFastAttn** - 通过利用扩散模型不同步骤之间的注意力冗余来减少计算量

## 使用方法

### 使用cache_execute.py直接比较多种缓存方法

这个脚本可以加载模型并应用不同的缓存方法进行比较：

```bash
python benchmark/cache_execute.py \
  --model_type pixart \
  --cache_methods None Fast Fb Tea \
  --num_inference_steps 20 \
  --height 512 \
  --width 512 \
  --output_dir cache_results
```

或者使用`All`选项测试所有方法：

```bash
python benchmark/cache_execute.py \
  --model_type pixart \
  --cache_methods All \
  --num_inference_steps 20
```

### 使用一键对比脚本

更简单的方法是使用一键对比脚本：

```bash
./examples/run_cache_comparison.sh pixart 20 512 512
```

参数说明：
1. 模型类型 (pixart, flux, sd3)，默认为pixart
2. 推理步数，默认为20
3. 图像高度，默认为512
4. 图像宽度，默认为512

结果将保存在`cache_comparison_results/[model_type]`目录下。

## 命令行参数

`cache_execute.py`支持以下参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model_type` | 模型类型 (`pixart`, `flux`, `sd3`) | `pixart` |
| `--model` | 模型路径或名称 | 基于`model_type`自动选择 |
| `--prompt` | 文本提示词 | `a beautiful landscape with mountains and a lake` |
| `--num_inference_steps` | 推理步数 | `30` |
| `--cache_methods` | 要测试的缓存方法 (`None`, `Fast`, `Fb`, `Tea`, `All`) | `All` |
| `--seed` | 随机种子 | `42` |
| `--height` | 图像高度 | `768` |
| `--width` | 图像宽度 | `768` |
| `--cache_ratio_threshold` | 缓存比率阈值 | `0.05` |
| `--motion_threshold` | FastCache运动阈值 | `0.1` |
| `--output_dir` | 结果输出目录 | `cache_execute_results` |

## 实现原理

该工具通过直接使用`apply_cache_on_transformer`函数应用不同的缓存方法，这个函数在`xfuser.model_executor.cache.diffusers_adapters.flux`模块中。不同的缓存方法由`use_cache`参数指定：

```python
apply_cache_on_transformer(
    transformer,
    rel_l1_thresh=threshold,
    return_hidden_states_first=False,
    num_steps=steps,
    use_cache="Fast",  # 或 "Fb", "Tea"
    motion_threshold=motion_threshold
)
```

这种方法确保所有缓存实现使用相同的API接口进行比较，提供公平的性能评估。 