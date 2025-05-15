import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 导入必要的模块
try:
    from diffusers import DiffusionPipeline
    print("Successfully imported DiffusionPipeline")
except ImportError as e:
    print(f"Error importing DiffusionPipeline: {e}")
    exit(1)

# 尝试导入FastCache相关模块
try:
    from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
    cache_available = True
    print("Successfully imported FastCache modules")
except ImportError as e:
    print(f"Warning: FastCache modules not available: {e}")
    cache_available = False

def apply_fastcache(model, threshold=0.15, motion_threshold=0.1, num_steps=20):
    """尝试对模型应用FastCache"""
    if not cache_available:
        print("FastCache模块不可用，跳过缓存应用")
        return model

    # 尝试找到transformer
    transformer = None
    
    if hasattr(model, "transformer"):
        transformer = model.transformer
        print("Found transformer in model.transformer")
    elif hasattr(model, "unet") and hasattr(model.unet, "transformer"):
        transformer = model.unet.transformer
        print("Found transformer in model.unet.transformer")
    elif hasattr(model, "text_to_video_unet") and hasattr(model.text_to_video_unet, "transformer"):
        transformer = model.text_to_video_unet.transformer
        print("Found transformer in model.text_to_video_unet.transformer")
    
    if transformer is None:
        print("找不到transformer组件，无法应用FastCache")
        return model
    
    try:
        # 应用FastCache
        apply_cache_on_transformer(
            transformer,
            rel_l1_thresh=threshold,
            return_hidden_states_first=False,
            num_steps=num_steps,
            use_cache="Fast",
            motion_threshold=motion_threshold
        )
        print(f"成功应用FastCache (threshold={threshold}, motion_threshold={motion_threshold})")
    except Exception as e:
        print(f"应用FastCache时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return model

def main():
    # 基本参数
    model_type = "stepvideo"  # 可选: "stepvideo", "cogvideox"
    
    if model_type == "stepvideo":
        model_id = "stepfun-ai/stepvideo-t2v"
    else:  # cogvideox
        model_id = "THUDM/CogVideoX1.5-5B"
    
    prompt = "a dog running in a field"
    num_frames = 8
    num_inference_steps = 20
    height = 256
    width = 256
    use_fastcache = True
    cache_threshold = 0.1
    motion_threshold = 0.1
    
    # 创建输出目录
    output_dir = "fastcache_video_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # 加载模型
    print(f"加载模型: {model_id}")
    start_time = time.time()
    
    try:
        # 使用通用DiffusionPipeline加载模型
        model = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        
        print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"模型类型: {type(model).__name__}")
        
        # 应用FastCache
        if use_fastcache:
            print("正在应用FastCache...")
            model = apply_fastcache(
                model,
                threshold=cache_threshold,
                motion_threshold=motion_threshold,
                num_steps=num_inference_steps
            )
        
        # 生成视频
        print(f"开始生成视频，提示词: '{prompt}'")
        start_time = time.time()
        
        result = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator
        )
        
        inference_time = time.time() - start_time
        print(f"视频生成完成，耗时: {inference_time:.2f}秒")
        
        # 保存帧为单独的图像
        print("正在保存视频帧...")
        frames = None
        
        # 检查不同的可能属性
        if hasattr(result, "frames"):
            frames = result.frames
            print("在result.frames中找到帧")
        elif hasattr(result, "videos") and len(result.videos) > 0:
            frames = result.videos[0]
            print("在result.videos[0]中找到帧")
        elif hasattr(result, "images"):
            frames = result.images
            print("在result.images中找到帧")
        
        if frames is None:
            print("在结果中找不到帧")
            print(f"结果属性: {dir(result)}")
            exit(1)
        
        # 保存每一帧
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                frame.save(frame_path)
                print(f"已保存第{i}帧至 {frame_path}")
        
        print(f"成功保存{len(frames)}帧至 {output_dir}")
        
        # 如果使用了FastCache，检查缓存命中情况
        if use_fastcache and hasattr(model, "transformer"):
            transformer = model.transformer
            for name, module in transformer.named_modules():
                if "FastCached" in str(type(module)):
                    if hasattr(module, "cache_hits") and hasattr(module, "total_steps"):
                        hits = module.cache_hits.item()
                        total = module.total_steps.item()
                        hit_ratio = hits / total if total > 0 else 0
                        print(f"FastCache统计: 命中={hits}, 总步数={total}, 命中率={hit_ratio:.2f}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 