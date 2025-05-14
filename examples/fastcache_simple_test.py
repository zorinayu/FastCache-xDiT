import os
import sys
import time
import torch
import argparse
from pathlib import Path
import json

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from diffusers import FluxPipeline, PixArtSigmaPipeline
    from PIL import Image
except ImportError:
    print("请确保已安装 diffusers>=0.30.0 和 Pillow")
    sys.exit(1)

class FastCacheAccelerator(torch.nn.Module):
    """FastCache加速器的简化实现"""
    
    def __init__(self, model, cache_ratio_threshold=0.05, motion_threshold=0.1):
        super().__init__()
        self.model = model
        self.cache_ratio_threshold = cache_ratio_threshold
        self.motion_threshold = motion_threshold
        
        # 缓存状态
        self.prev_hidden_states = None
        self.cache_hits = 0
        self.total_steps = 0
        
        # 对于静态token的线性近似
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        else:
            # 尝试从参数估计隐藏大小
            for param in model.parameters():
                if len(param.shape) > 1:
                    hidden_size = param.shape[-1]
                    break
            else:
                hidden_size = 768  # 默认值

        self.cache_projection = torch.nn.Linear(hidden_size, hidden_size).to(model.device)
        
        print(f"FastCache初始化：threshold={cache_ratio_threshold}, motion={motion_threshold}")
    
    def compute_relative_change(self, current, previous):
        """计算当前和前一个隐藏状态之间的相对变化"""
        if previous is None:
            return float('inf')
            
        # 计算差异的Frobenius范数
        diff_norm = torch.norm(current - previous, p='fro')
        prev_norm = torch.norm(previous, p='fro')
        
        # 避免除以零
        if prev_norm == 0:
            return float('inf')
            
        return (diff_norm / prev_norm).item()
    
    def should_use_cache(self, hidden_states):
        """确定是否应该使用缓存状态"""
        if self.prev_hidden_states is None:
            return False
            
        # 计算相对变化
        delta = self.compute_relative_change(hidden_states, self.prev_hidden_states)
        
        # 与阈值比较
        return delta <= self.cache_ratio_threshold
    
    def compute_motion_saliency(self, hidden_states):
        """计算每个token的运动显著性"""
        if self.prev_hidden_states is None:
            return torch.ones(hidden_states.shape[1], device=hidden_states.device)
            
        # 计算token级别的差异
        token_diffs = (hidden_states - self.prev_hidden_states).abs()
        
        # 取特征维度上的最大值以获取token显著性
        token_saliency = token_diffs.max(dim=-1)[0].squeeze(0)
        
        # 归一化显著性
        if token_saliency.max() > 0:
            token_saliency = token_saliency / token_saliency.max()
            
        return token_saliency
    
    def forward(self, hidden_states, **kwargs):
        """
        通过FastCache处理隐藏状态
        """
        self.total_steps += 1
        
        # 如果是第一步，正常执行
        if self.prev_hidden_states is None:
            output = self.model(hidden_states, **kwargs)
            self.prev_hidden_states = hidden_states.detach().clone()
            return output
        
        # 分析运动并决定缓存策略
        if self.should_use_cache(hidden_states):
            # 缓存命中 - 重用先前的状态
            self.cache_hits += 1
            
            # 应用线性投影而不是完整的transformer
            output = self.cache_projection(hidden_states)
            return output
        
        # 计算token的运动显著性
        motion_saliency = self.compute_motion_saliency(hidden_states)
        motion_mask = motion_saliency > self.motion_threshold
        
        # 如果检测到显著运动，正常处理
        if motion_mask.sum() / motion_mask.numel() > 0.5:
            output = self.model(hidden_states, **kwargs)
        else:
            # 将token分为运动token和静态token
            batch_size, seq_len, hidden_dim = hidden_states.shape
            motion_indices = torch.where(motion_mask)[0]
            static_indices = torch.where(~motion_mask)[0]
            
            if len(motion_indices) > 0:
                # 通过完整的transformer处理运动token
                motion_states = hidden_states.index_select(1, motion_indices)
                motion_output = self.model(motion_states, **kwargs)
                
                # 通过线性投影处理静态token
                static_states = hidden_states.index_select(1, static_indices)
                static_output = self.cache_projection(static_states)
                
                # 合并输出
                output = hidden_states.clone()
                output.index_copy_(1, motion_indices, motion_output)
                output.index_copy_(1, static_indices, static_output)
            else:
                # 所有token都是静态的，使用线性投影
                output = self.cache_projection(hidden_states)
        
        # 更新缓存
        self.prev_hidden_states = hidden_states.detach().clone()
        
        return output
    
    def get_cache_hit_ratio(self):
        """返回缓存命中率"""
        if self.total_steps == 0:
            return 0.0
        return self.cache_hits / self.total_steps

def apply_fastcache_to_transformer(model, cache_threshold=0.05, motion_threshold=0.1):
    """将FastCache应用于transformer模块"""
    accelerators = []
    
    def apply_recursive(module, prefix=''):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 检查模块类型
            module_type = child.__class__.__name__
            if "Transformer" in module_type or "Attention" in module_type:
                # 创建加速器
                accelerator = FastCacheAccelerator(
                    child, 
                    cache_ratio_threshold=cache_threshold,
                    motion_threshold=motion_threshold
                )
                accelerators.append((full_name, accelerator))
                
                # 替换原始模块
                setattr(module, name, accelerator)
                print(f"已应用FastCache到 {module_type} 在 {full_name}")
            else:
                # 递归处理子模块
                apply_recursive(child, full_name)
    
    # 查找并应用FastCache
    if hasattr(model, "transformer"):
        apply_recursive(model.transformer)
    elif hasattr(model, "unet"):
        apply_recursive(model.unet)
    
    return model, accelerators

def parse_args():
    parser = argparse.ArgumentParser(description="FastCache简单测试脚本")
    parser.add_argument("--model_type", type=str, choices=["pixart", "flux"], default="pixart", help="模型类型")
    parser.add_argument("--model", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", help="模型ID")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on the moon", help="生成提示")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="推理步骤数")
    parser.add_argument("--height", type=int, default=768, help="图像高度")
    parser.add_argument("--width", type=int, default=768, help="图像宽度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache_threshold", type=float, default=0.05, help="缓存比率阈值")
    parser.add_argument("--motion_threshold", type=float, default=0.1, help="运动阈值")
    parser.add_argument("--output_dir", type=str, default="fastcache_results", help="输出目录")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print(f"测试模型类型: {args.model_type}")
    print(f"测试模型: {args.model}")
    
    try:
        # 加载模型
        print(f"正在加载模型...")
        if args.model_type == "pixart":
            model = PixArtSigmaPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif args.model_type == "flux":
            model = FluxPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            raise ValueError(f"不支持的模型类型: {args.model_type}")
            
        # 基线测试
        print(f"\n执行基线测试 ({args.num_inference_steps} 步)...")
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        
        baseline_start_time = time.time()
        with torch.no_grad():
            baseline_result = model(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                generator=generator,
            )
        baseline_time = time.time() - baseline_start_time
        print(f"基线完成，耗时 {baseline_time:.2f} 秒")
        
        # 保存基线图像
        baseline_image = baseline_result.images[0]
        baseline_path = os.path.join(args.output_dir, "baseline_image.png")
        baseline_image.save(baseline_path)
        print(f"基线图像已保存到 {baseline_path}")
        
        # 应用FastCache
        print(f"\n应用FastCache加速...")
        model, accelerators = apply_fastcache_to_transformer(
            model,
            cache_threshold=args.cache_threshold,
            motion_threshold=args.motion_threshold
        )
        
        # FastCache测试
        print(f"\n执行FastCache测试 ({args.num_inference_steps} 步)...")
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        
        fastcache_start_time = time.time()
        with torch.no_grad():
            fastcache_result = model(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                generator=generator,
            )
        fastcache_time = time.time() - fastcache_start_time
        print(f"FastCache完成，耗时 {fastcache_time:.2f} 秒")
        
        # 保存FastCache图像
        fastcache_image = fastcache_result.images[0]
        fastcache_path = os.path.join(args.output_dir, "fastcache_image.png")
        fastcache_image.save(fastcache_path)
        print(f"FastCache图像已保存到 {fastcache_path}")
        
        # 计算加速比
        speedup = baseline_time / fastcache_time
        print(f"\nFastCache加速比: {speedup:.2f}x")
        
        # 显示缓存命中统计
        total_hits = sum(acc.cache_hits for _, acc in accelerators)
        total_steps = sum(acc.total_steps for _, acc in accelerators)
        if total_steps > 0:
            hit_ratio = total_hits / total_steps
            print(f"缓存命中率: {hit_ratio:.2%} ({total_hits}/{total_steps})")
        
        # 保存结果
        stats = {
            "model_type": args.model_type,
            "model": args.model,
            "prompt": args.prompt,
            "steps": args.num_inference_steps,
            "resolution": f"{args.height}x{args.width}",
            "seed": args.seed,
            "cache_threshold": args.cache_threshold,
            "motion_threshold": args.motion_threshold,
            "baseline_time": baseline_time,
            "fastcache_time": fastcache_time,
            "speedup": speedup,
            "cache_hits": int(total_hits),
            "total_steps": int(total_steps),
            "hit_ratio": float(hit_ratio) if total_steps > 0 else 0.0
        }
        
        stats_path = os.path.join(args.output_dir, "results.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n结果已保存到 {args.output_dir}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 