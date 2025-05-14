import argparse
import os
import time
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 尝试导入必要的模块
try:
    from diffusers import StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Warning: diffusers库未正确安装，请确保安装了diffusers>=0.30.0")

# 从项目中导入必要的模块
from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from xfuser.logger import init_logger

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FastCache Benchmark (Simple Version)")
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
    parser.add_argument(
        "--height", type=int, default=768, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=768, help="Image width"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fastcache_results",
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
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """设置随机种子以便结果可重现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args):
    """加载扩散模型"""
    print(f"Loading {args.model_type} model: {args.model}")
    
    if args.model_type == "sd3":
        try:
            model = StableDiffusion3Pipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
            return model
        except Exception as e:
            print(f"Error loading SD3 model: {e}")
            return None
    elif args.model_type == "flux":
        try:
            model = FluxPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
            return model
        except Exception as e:
            print(f"Error loading Flux model: {e}")
            return None
    else:
        print(f"Unsupported model type: {args.model_type}")
        return None


def run_with_fastcache(model, args):
    """使用FastCache运行模型"""
    if model is None:
        print("No model loaded, cannot run FastCache")
        return None, float('inf')
        
    set_seed(args.seed)
    
    try:
        # 创建FastCache包装器
        print("Creating FastCache wrapper...")
        fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
        
        # 启用FastCache
        print(f"Enabling FastCache with threshold={args.cache_ratio_threshold}, motion={args.motion_threshold}")
        fastcache_wrapper.enable_fastcache(
            cache_ratio_threshold=args.cache_ratio_threshold,
            motion_threshold=args.motion_threshold,
        )
        
        # 运行推理
        print("Running inference with FastCache...")
        start_time = time.time()
        result = fastcache_wrapper(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 获取缓存统计信息
        stats = fastcache_wrapper.get_cache_statistics()
        print("\nFastCache statistics:")
        total_hit_ratio = 0.0
        count = 0
        for name, stat in stats.items():
            hit_ratio = stat['cache_hit_ratio']
            total_hit_ratio += hit_ratio
            count += 1
            print(f"  {name}: {hit_ratio:.2%} hit ratio")
        
        if count > 0:
            avg_hit_ratio = total_hit_ratio / count
            print(f"Average hit ratio: {avg_hit_ratio:.2%}")
        
        return result, elapsed
    except Exception as e:
        print(f"Error running FastCache: {e}")
        import traceback
        traceback.print_exc()
        return None, float('inf')


def save_results(result, elapsed, args):
    """保存结果"""
    if result is None:
        print("No results to save")
        return
        
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存时间信息
        with open(os.path.join(args.output_dir, "timing.txt"), "w") as f:
            f.write(f"FastCache: {elapsed:.4f}s\n")
        
        # 保存图像
        if hasattr(result, 'images') and len(result.images) > 0:
            image_path = os.path.join(args.output_dir, "fastcache_result.png")
            result.images[0].save(image_path)
            print(f"Result image saved to {image_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 显示参数
    print("FastCache Benchmark (Simple Version)")
    print("----------------------------------")
    print(f"Model type: {args.model_type}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.num_inference_steps}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Cache threshold: {args.cache_ratio_threshold}")
    print(f"Motion threshold: {args.motion_threshold}")
    print("----------------------------------")
    
    # 加载模型
    model = load_model(args)
    if model is None:
        return
    
    # 运行FastCache
    print("\nRunning with FastCache...")
    result, elapsed = run_with_fastcache(model, args)
    
    # 显示结果
    if result is not None:
        print(f"\nFastCache completed in {elapsed:.4f}s")
        
        # 保存结果
        save_results(result, elapsed, args)
        print(f"\nResults saved to {args.output_dir}")
    else:
        print("\nFailed to run FastCache")


if __name__ == "__main__":
    main() 