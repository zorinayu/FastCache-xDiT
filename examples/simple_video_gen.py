import os
import time
import torch
import numpy as np
from PIL import Image
import imageio
from pathlib import Path

# 尝试导入StepVideoGenerationPipeline
try:
    from diffusers import StepVideoGenerationPipeline
    print("Successfully imported StepVideoGenerationPipeline")
except ImportError as e:
    print(f"Error importing StepVideoGenerationPipeline: {e}")
    exit(1)

def save_video(frames, path, fps=8):
    """保存PIL图像列表为视频"""
    print(f"Saving {len(frames)} frames to {path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 转换PIL图像为numpy数组
    if isinstance(frames[0], Image.Image):
        numpy_frames = [np.array(frame) for frame in frames]
        frames = np.stack(numpy_frames)
    
    # 保存为视频
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved to {path}")
    return True

def main():
    # 设置参数
    model_id = "stepfun-ai/stepvideo-t2v"
    prompt = "a dog running in a field"
    num_frames = 8
    height = 256
    width = 256
    num_inference_steps = 20
    seed = 42
    
    # 设置输出路径
    output_dir = "video_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading model: {model_id}")
    start_time = time.time()
    
    # 加载模型
    pipeline = StepVideoGenerationPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # 设置随机生成器
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # 运行推理
    print(f"Generating video with prompt: '{prompt}'")
    start_time = time.time()
    result = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator
    )
    inference_time = time.time() - start_time
    print(f"Video generated in {inference_time:.2f} seconds")
    
    # 获取视频帧
    frames = None
    if hasattr(result, "frames"):
        frames = result.frames
    elif hasattr(result, "videos") and len(result.videos) > 0:
        frames = result.videos[0]
    elif hasattr(result, "images"):
        frames = result.images
    
    if frames is None:
        print("No frames found in result")
        exit(1)
    
    # 保存视频
    video_path = os.path.join(output_dir, "simple_video.mp4")
    save_video(frames, video_path)
    
    print(f"Process completed. Video saved to {video_path}")

if __name__ == "__main__":
    main() 