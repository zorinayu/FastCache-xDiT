import os
import sys
import time
import torch
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Try to import necessary modules
try:
    from diffusers import (
        StepVideoGenerationPipeline,
        CogVideoX15Pipeline,
        ConsisIDPipeline,
    )
except ImportError:
    print("Warning: diffusers library not properly installed or video models not available")

# 导入cache相关函数
try:
    # 直接从flux.py导入缓存应用函数
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
    cache_modules_available = True
    print("Successfully imported cache modules from xfuser")
except ImportError as e:
    print(f"Warning: xfuser cache modules not found: {e}")
    cache_modules_available = False

def parse_args():
    parser = argparse.ArgumentParser(description="Video Cache Execution Test")
    parser.add_argument("--model_type", type=str, 
                       choices=["stepvideo", "cogvideox", "consisid"], 
                       default="stepvideo")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a dog running in a field")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_methods", type=str, nargs="+", 
                        choices=["None", "Fast", "Fb", "Tea", "All"], 
                        default=["All"])
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.15)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="video_cache_results")
    
    args = parser.parse_args()
    
    # Set default model based on model_type if not provided
    if args.model is None:
        if args.model_type == "stepvideo":
            args.model = "stepfun-ai/stepvideo-t2v"
        elif args.model_type == "cogvideox":
            args.model = "THUDM/CogVideoX1.5-5B"
        elif args.model_type == "consisid":
            args.model = "PKU-YuanGroup/ConsisID"
    
    # 如果选择了All，则测试所有缓存方法
    if "All" in args.cache_methods:
        args.cache_methods = ["None", "Fast", "Fb", "Tea"]
    
    return args

def apply_cache_to_model(model, cache_method, rel_l1_thresh=0.15, num_steps=20, motion_threshold=0.1):
    """直接使用apply_cache_on_transformer函数应用缓存方法"""
    if not cache_modules_available:
        print(f"Warning: Cache modules not available, cannot apply {cache_method}")
        return model
    
    if not hasattr(model, "transformer"):
        # 尝试找到transformer组件
        if hasattr(model, "unet") and hasattr(model.unet, "transformer"):
            transformer = model.unet.transformer
        elif hasattr(model, "text_to_video_unet") and hasattr(model.text_to_video_unet, "transformer"):
            transformer = model.text_to_video_unet.transformer
        else:
            print(f"Warning: Cannot find transformer component in model, cannot apply {cache_method}")
            return model
    else:
        transformer = model.transformer
    
    if cache_method == "None":
        # 不应用缓存
        return model
    
    try:
        # 直接使用flux.py中的函数
        apply_cache_on_transformer(
            transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=False,
            num_steps=num_steps,
            use_cache=cache_method,  # "Fast", "Fb", or "Tea"
            motion_threshold=motion_threshold
        )
        print(f"Successfully applied {cache_method}Cache to transformer")
    except Exception as e:
        print(f"Error applying {cache_method}Cache: {e}")
    
    return model

def run_inference(model, args, generator):
    """运行视频推理并计时"""
    start_time = time.time()
    
    with torch.no_grad():
        # 尝试适配不同的视频生成模型API
        try:
            if args.model_type == "stepvideo":
                result = model(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    generator=generator,
                )
            elif args.model_type == "cogvideox":
                result = model(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    generator=generator,
                )
            elif args.model_type == "consisid":
                result = model(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    generator=generator,
                )
            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")
        except Exception as e:
            print(f"Error during inference: {e}")
            # 尝试通用方法
            result = model(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                generator=generator,
            )
    
    inference_time = time.time() - start_time
    
    return result, inference_time

def load_video_model(args):
    """加载视频生成模型"""
    try:
        if args.model_type == "stepvideo":
            model = StepVideoGenerationPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "cogvideox":
            model = CogVideoX15Pipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "consisid":
            model = ConsisIDPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to generic diffusers import...")
        
        # 尝试通用导入方法
        from diffusers import DiffusionPipeline
        model = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to("cuda")
        return model

def save_video(frames, path, fps=8):
    """保存帧序列为视频文件"""
    try:
        if isinstance(frames, list) and isinstance(frames[0], Image.Image):
            # 如果是PIL图像列表，转换为numpy数组
            frames = [np.array(frame) for frame in frames]
        
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            # 已经是NHWC格式的numpy数组
            pass
        elif isinstance(frames, torch.Tensor):
            # 如果是torch.Tensor，转换为numpy
            frames = frames.cpu().numpy()
            if frames.shape[1] == 3:  # NCHW -> NHWC
                frames = np.transpose(frames, (0, 2, 3, 1))
        
        # 确保像素值在[0, 255]范围内且为uint8类型
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
        
        # 保存视频
        imageio.mimsave(path, frames, fps=fps)
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        # 尝试逐帧保存为图像
        os.makedirs(path.replace('.mp4', ''), exist_ok=True)
        for i, frame in enumerate(frames):
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if isinstance(frame, np.ndarray):
                if frame.shape[0] == 3:  # CHW
                    frame = np.transpose(frame, (1, 2, 0))
                frame = Image.fromarray(frame.astype(np.uint8))
            if isinstance(frame, Image.Image):
                frame.save(f"{path.replace('.mp4', '')}/frame_{i:04d}.png")
        return False

def main():
    args = parse_args()
    print(f"Testing video cache methods for {args.model} ({args.model_type})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 存储所有结果
    results = {}
    
    # 为每个缓存方法加载新的模型实例并运行推理
    for method in args.cache_methods:
        print(f"\n===== Testing {method}Cache =====")
        method_name = "Baseline" if method == "None" else f"{method}Cache"
        
        # 加载模型
        try:
            model = load_video_model(args)
            
            # 应用缓存方法
            model = apply_cache_to_model(
                model,
                cache_method=method,
                rel_l1_thresh=args.cache_ratio_threshold,
                num_steps=args.num_inference_steps,
                motion_threshold=args.motion_threshold
            )
            
            # 创建固定的随机生成器以确保结果一致
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            
            # 运行推理
            print(f"Running inference with {method_name} ({args.num_inference_steps} steps)...")
            result, inference_time = run_inference(model, args, generator)
            print(f"{method_name} inference completed in {inference_time:.2f} seconds")
            
            # 提取视频帧
            if hasattr(result, "frames"):
                frames = result.frames
            elif hasattr(result, "videos"):
                frames = result.videos[0]
            elif hasattr(result, "images"):
                frames = result.images
            else:
                # 尝试直接使用结果
                frames = result
            
            # 保存视频
            video_path = os.path.join(args.output_dir, f"{method.lower()}_video.mp4")
            success = save_video(frames, video_path, fps=args.fps)
            if success:
                print(f"Video saved to {video_path}")
            else:
                print(f"Failed to save video, individual frames saved to {video_path.replace('.mp4', '/')}")
            
            # 收集统计信息
            results[method] = {
                "method": method_name,
                "model": args.model,
                "model_type": args.model_type,
                "prompt": args.prompt,
                "steps": args.num_inference_steps,
                "frames": args.num_frames,
                "resolution": f"{args.height}x{args.width}",
                "seed": args.seed,
                "inference_time": inference_time,
                "cache_threshold": args.cache_ratio_threshold
            }
            
            # 为FastCache方法添加motion_threshold
            if method == "Fast":
                results[method]["motion_threshold"] = args.motion_threshold
                
        except Exception as e:
            print(f"Error processing {method}Cache: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {
                "method": method_name,
                "model": args.model,
                "model_type": args.model_type,
                "error": str(e)
            }
    
    # 保存整体测试结果
    results_path = os.path.join(args.output_dir, "video_cache_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to {results_path}")
    
    # 打印比较摘要
    baseline_time = results.get("None", {}).get("inference_time", 0)
    if baseline_time > 0:
        print("\n===== Cache Methods Comparison =====")
        print(f"{'Method':<15} {'Time (s)':<10} {'Speedup':<10}")
        print("-" * 35)
        
        for method in args.cache_methods:
            if "inference_time" in results.get(method, {}):
                method_time = results[method]["inference_time"]
                speedup = baseline_time / method_time if method != "None" else 1.0
                print(f"{results[method]['method']:<15} {method_time:<10.2f} {speedup:<10.2f}x")

if __name__ == "__main__":
    main() 