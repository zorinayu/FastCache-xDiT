import os
import time
import torch
from PIL import Image

# 尝试导入CogVideoXPipeline
try:
    from diffusers import DiffusionPipeline
    print("Successfully imported DiffusionPipeline")
except ImportError as e:
    print(f"Error importing DiffusionPipeline: {e}")
    exit(1)

def main():
    # 基本参数
    model_id = "THUDM/CogVideoX1.5-5B"
    prompt = "a dog running in a field"
    num_frames = 8
    num_inference_steps = 20
    height = 256
    width = 256
    
    # 创建输出目录
    output_dir = "cogvideo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"Loading model: {model_id}")
    start_time = time.time()
    
    try:
        # 使用通用DiffusionPipeline加载模型
        model = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"Model type: {type(model).__name__}")
        
        # 生成视频
        print(f"Generating video for prompt: '{prompt}'")
        start_time = time.time()
        
        result = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        
        print(f"Video generated in {time.time() - start_time:.2f} seconds")
        
        # 保存帧为单独的图像
        print("Saving frames as individual images")
        frames = None
        
        # 检查不同的可能属性
        if hasattr(result, "frames"):
            frames = result.frames
            print("Found frames in result.frames")
        elif hasattr(result, "videos") and len(result.videos) > 0:
            frames = result.videos[0]
            print("Found frames in result.videos[0]")
        elif hasattr(result, "images"):
            frames = result.images
            print("Found frames in result.images")
        
        if frames is None:
            print("Couldn't find frames in the result")
            print(f"Result attributes: {dir(result)}")
            exit(1)
        
        # 保存每一帧
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                frame.save(frame_path)
                print(f"Saved frame {i} to {frame_path}")
        
        print(f"Successfully saved {len(frames)} frames to {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 