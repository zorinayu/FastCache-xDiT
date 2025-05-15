import logging
import os
import time
import torch
import argparse
from pathlib import Path

# First try importing base diffusers components that don't involve circular references
try:
    from diffusers.utils import export_to_video
except ImportError as e:
    print(f"Error importing diffusers utilities: {e}")
    exit(1)

# Import xfuser components in the correct order to avoid circular dependencies
try:
    # Import xfuser base components first
    from xfuser.core.distributed import get_world_group, get_runtime_state
    from xfuser.config import FlexibleArgumentParser, EngineConfig

    # Now import the pipeline wrapper
    from xfuser import xFuserCogVideoXPipeline
except ImportError as e:
    print(f"Error importing xfuser components: {e}")
    print("Make sure xfuser package is installed correctly.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Generation with xFuser")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF repo ID")
    parser.add_argument("--prompt", type=str, default="A little girl is riding a bicycle at high speed. Focused, detailed, realistic.", 
                      help="Text prompt for video generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--num_frames", type=int, default=17, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=768, help="Video height")
    parser.add_argument("--width", type=int, default=1360, help="Video width") 
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--enable_tiling", action="store_true", help="Enable tiling for VAE")
    parser.add_argument("--enable_slicing", action="store_true", help="Enable slicing for VAE")
    
    # Add additional arguments for xfuser configuration
    parser.add_argument("--use_cfg_parallel", action="store_true", help="Use classifier-free guidance parallel")
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses attention degree")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring attention degree")
    
    # Add local rank for distributed execution
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    args = parse_args()
    logger = setup_logging()
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Create xfuser engine config
    xfuser_args = FlexibleArgumentParser().parse_args([
        "--model", args.model,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--prompt", args.prompt,
        "--num_inference_steps", str(args.num_inference_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--ulysses_degree", str(args.ulysses_degree),
        "--ring_degree", str(args.ring_degree),
    ])
    
    if args.use_cfg_parallel:
        xfuser_args.use_cfg_parallel = True
    
    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    
    # Set local device and seed
    local_rank = args.local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed)
    
    # Load model with xfuser wrapper
    logger.info(f"Loading CogVideoX model from {args.model}")
    start_load_time = time.time()
    
    try:
        # Initialize the model with xfuser
        pipe = xFuserCogVideoXPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
        ).to(device)
        
        logger.info(f"Model loaded in {time.time() - start_load_time:.2f} seconds")
        
        # Apply optimizations if requested
        if args.enable_tiling:
            logger.info("Enabling VAE tiling")
            pipe.vae.enable_tiling()
        
        if args.enable_slicing:
            logger.info("Enabling VAE slicing")
            pipe.vae.enable_slicing()
        
        # Generate video
        logger.info(f"Generating video with {args.num_inference_steps} steps")
        logger.info(f"Prompt: '{args.prompt}'")
        logger.info(f"Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")
        
        # Reset CUDA memory stats before inference
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        generator = torch.Generator(device=device).manual_seed(args.seed)
        output = pipe(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        
        inference_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # Convert to GB
        
        logger.info(f"Generation completed in {inference_time:.2f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} GB")
        
        # Save generated video if this rank has the output
        if pipe.is_dp_last_group():
            parallel_info = (
                f"cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{args.ulysses_degree}_ring{args.ring_degree}"
            )
            output_filename = f"results/cogvideox_{parallel_info}_{args.width}x{args.height}.mp4"
            export_to_video(output.frames[0], output_filename, fps=8)
            logger.info(f"Video saved to {output_filename}")
        
        # Clean up
        get_runtime_state().destroy_distributed_env()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are correctly installed")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        logger.error("Check GPU memory and model compatibility")
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()

# Ensure the import error for xFuserArgs is properly handled
try:
    from xfuser import xFuserArgs
except ImportError as e:
    # If xFuserArgs is not found, define a simpler implementation for parsing args
    class xFuserArgs:
        @staticmethod
        def from_cli_args(args):
            from xfuser.config import xFuserArgs
            return xFuserArgs.from_cli_args(args)

if __name__ == "__main__":
    main() 