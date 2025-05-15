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

# Try to import necessary diffusers classes directly without going through xfuser imports
try:
    # Import directly from diffusers
    from diffusers import (
        DiffusionPipeline,
        StepVideoGenerationPipeline,
    )
    # Try to import CogVideoX and ConsisID separately since they might be missing in some versions
    try:
        from diffusers import CogVideoXPipeline
    except ImportError:
        print("CogVideoXPipeline not available in this diffusers version")
        
    try:
        from diffusers import ConsisIDPipeline
    except ImportError:
        print("ConsisIDPipeline not available in this diffusers version")
except ImportError:
    print("Warning: diffusers library not properly installed")

# 导入cache相关函数，避免通过pipeline模块导入
try:
    # 直接导入FastCache的实现，避免循环导入
    from xfuser.model_executor.cache.utils import (
        FastCachedTransformerBlocks,
        TeaCachedTransformerBlocks, 
        FBCachedTransformerBlocks
    )
    
    # 导入应用缓存的函数
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
    
    if cache_method == "None":
        # 不应用缓存
        return model
    
    # 查找模型的transformer组件
    transformer = None
    if hasattr(model, "transformer"):
        transformer = model.transformer
    elif hasattr(model, "unet") and hasattr(model.unet, "transformer"):
        transformer = model.unet.transformer
    elif hasattr(model, "text_to_video_unet") and hasattr(model.text_to_video_unet, "transformer"):
        transformer = model.text_to_video_unet.transformer
    
    if transformer is None:
        print(f"Warning: Cannot find transformer component in model, cannot apply {cache_method}")
        return model
    
    try:
        # 应用缓存
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
            return model
        elif args.model_type == "cogvideox" and 'CogVideoXPipeline' in globals():
            model = CogVideoXPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
            return model
        elif args.model_type == "consisid" and 'ConsisIDPipeline' in globals():
            model = ConsisIDPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
            return model
        else:
            print(f"Model type {args.model_type} not directly supported, trying generic pipeline")
            # Fallback to generic pipeline
            model = DiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to generic diffusers import...")
        
        # 尝试通用导入方法
        model = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to("cuda")
        return model

def save_video(frames, path, fps=8):
    """保存帧序列为视频文件"""
    try:
        # 首先确保我们有一个有效的帧列表
        if frames is None:
            raise ValueError("Frames cannot be None")
            
        # 处理可能的帧格式
        if hasattr(frames, "images"):
            frames = frames.images
        elif hasattr(frames, "frames"):
            frames = frames.frames
        elif hasattr(frames, "videos") and isinstance(frames.videos, list) and len(frames.videos) > 0:
            frames = frames.videos[0]
        
        # 处理嵌套列表结构
        if isinstance(frames, list):
            # 打印更多调试信息
            print(f"Frames type: {type(frames)}")
            if len(frames) > 0:
                print(f"First element type: {type(frames[0])}")
                # 处理列表的列表情况
                if isinstance(frames[0], list):
                    print("Detected nested list structure, flattening...")
                    flat_frames = []
                    for sublist in frames:
                        if isinstance(sublist, list):
                            flat_frames.extend(sublist)
                        else:
                            flat_frames.append(sublist)
                    frames = flat_frames
                    print(f"Flattened frames length: {len(frames)}")
                    if len(frames) > 0:
                        print(f"First flattened element type: {type(frames[0])}")
            
            # 如果是PIL图像列表，转换为numpy数组并堆叠
            if len(frames) == 0:
                raise ValueError("Empty frames list")
                
            if all(isinstance(frame, Image.Image) for frame in frames):
                # 转换PIL图像列表为numpy数组
                numpy_frames = []
                for frame in frames:
                    numpy_frame = np.array(frame)
                    if numpy_frame.ndim == 2:  # 灰度图像
                        numpy_frame = np.stack([numpy_frame] * 3, axis=-1)
                    numpy_frames.append(numpy_frame)
                frames = np.stack(numpy_frames)
                print(f"Converted PIL images to numpy array with shape: {frames.shape}")
            elif all(isinstance(frame, np.ndarray) for frame in frames):
                # 已经是numpy数组列表，直接堆叠
                try:
                    frames = np.stack(frames)
                    print(f"Stacked numpy arrays with shape: {frames.shape}")
                except Exception as stack_error:
                    print(f"Error stacking numpy arrays: {stack_error}")
                    # 尝试确保所有数组具有相同的形状
                    first_shape = frames[0].shape
                    filtered_frames = [f for f in frames if f.shape == first_shape]
                    if len(filtered_frames) > 0:
                        frames = np.stack(filtered_frames)
                        print(f"Stacked {len(filtered_frames)} compatible numpy arrays with shape: {frames.shape}")
                    else:
                        raise ValueError("Could not find compatible frames to stack")
                        
        # 如果是torch.Tensor，转换为numpy
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
            if frames.ndim == 4 and frames.shape[1] == 3:  # NCHW -> NHWC
                frames = np.transpose(frames, (0, 2, 3, 1))
            print(f"Converted torch tensor to numpy array with shape: {frames.shape}")
                
        # 确保frames是numpy数组且具有正确的维度
        if not isinstance(frames, np.ndarray):
            # 如果转换失败，尝试保存个别帧
            print(f"Could not convert frames to numpy array, will save individual frames")
            raise TypeError(f"Failed to convert frames to numpy array, got {type(frames)}")
        
        if frames.ndim != 4:
            print(f"Expected 4D array (frames, height, width, channels), got shape {frames.shape}")
            # 尝试修复维度问题
            if frames.ndim == 3:
                if frames.shape[0] == 3:  # 可能是单个CHW图像
                    frames = np.transpose(frames, (1, 2, 0))  # HWC
                    frames = frames[np.newaxis, ...]  # 添加批次维度
                    print(f"Reshaped to 4D array with shape: {frames.shape}")
                else:  # 假设是HWC格式的单个图像
                    frames = frames[np.newaxis, ...]  # 添加批次维度
                    print(f"Added batch dimension, new shape: {frames.shape}")
            else:
                raise ValueError(f"Cannot reshape {frames.shape} to 4D array")
            
        # 确保像素值在[0, 255]范围内且为uint8类型
        if frames.dtype != np.uint8:
            if np.issubdtype(frames.dtype, np.floating):
                # 处理NaN值
                frames = np.nan_to_num(frames, nan=0.0)
                if frames.max() <= 1.0:
                    frames = (frames * 255).round().astype(np.uint8)
                else:
                    frames = frames.round().astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
            print(f"Converted to uint8 array with shape: {frames.shape}")
        
        # 保存视频
        print(f"Saving {len(frames)} frames to video at {path}")
        imageio.mimsave(path, frames, fps=fps)
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        # 尝试逐帧保存为图像（作为备份方法）
        save_dir = path.replace('.mp4', '')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory for individual frames: {save_dir}")
        
        # 尝试获取可以保存的帧
        saveable_frames = frames
        if not isinstance(saveable_frames, (list, tuple, np.ndarray)):
            if hasattr(saveable_frames, "images"):
                saveable_frames = saveable_frames.images
            elif hasattr(saveable_frames, "frames"):
                saveable_frames = saveable_frames.frames
            else:
                print(f"Cannot save individual frames: unknown type {type(saveable_frames)}")
                return False
        
        # 处理嵌套列表
        if isinstance(saveable_frames, list) and len(saveable_frames) > 0 and isinstance(saveable_frames[0], list):
            flat_frames = []
            for sublist in saveable_frames:
                if isinstance(sublist, list):
                    flat_frames.extend(sublist)
                else:
                    flat_frames.append(sublist)
            saveable_frames = flat_frames
        
        # 逐帧保存
        success_count = 0
        for i, frame in enumerate(saveable_frames):
            try:
                # 检查帧的类型
                print(f"Frame {i} type: {type(frame)}")
                
                # 确保每一帧是PIL图像
                frame_pil = None
                if isinstance(frame, Image.Image):
                    frame_pil = frame
                elif isinstance(frame, torch.Tensor):
                    frame_np = frame.cpu().numpy()
                    # 处理不同的形状
                    if frame_np.ndim == 3 and frame_np.shape[0] == 3:  # CHW
                        frame_np = np.transpose(frame_np, (1, 2, 0))
                    elif frame_np.ndim == 4:  # NCHW or NHWC
                        if frame_np.shape[1] == 3:  # NCHW
                            frame_np = np.transpose(frame_np[0], (1, 2, 0))
                        else:  # NHWC
                            frame_np = frame_np[0]
                    # 确保是uint8格式
                    if frame_np.dtype != np.uint8:
                        frame_np = np.nan_to_num(frame_np, nan=0.0)
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).round().astype(np.uint8)
                        else:
                            frame_np = frame_np.round().astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                elif isinstance(frame, np.ndarray):
                    # 处理不同的形状
                    if frame.ndim == 3 and frame.shape[0] == 3:  # CHW
                        frame = np.transpose(frame, (1, 2, 0))
                    # 确保是uint8格式
                    if frame.dtype != np.uint8:
                        frame = np.nan_to_num(frame, nan=0.0)
                        if frame.max() <= 1.0:
                            frame = (frame * 255).round().astype(np.uint8)
                        else:
                            frame = frame.round().astype(np.uint8)
                    frame_pil = Image.fromarray(frame)
                else:
                    print(f"Skipping frame {i}: unsupported type {type(frame)}")
                    continue
                
                if frame_pil:
                    frame_path = f"{save_dir}/frame_{i:04d}.png"
                    frame_pil.save(frame_path)
                    success_count += 1
                else:
                    print(f"Could not convert frame {i} to PIL image")
            except Exception as frame_e:
                print(f"Error saving frame {i}: {frame_e}")
        
        print(f"Successfully saved {success_count} individual frames")
        return success_count > 0

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