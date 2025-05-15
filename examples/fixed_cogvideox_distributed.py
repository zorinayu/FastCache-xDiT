import logging
import os
import time
import torch

# First try importing base components that don't involve circular references
try:
    from diffusers.utils import export_to_video
except ImportError as e:
    print(f"Error importing diffusers utilities: {e}")
    exit(1)

# Import xfuser components in the correct order to avoid circular dependencies
try:
    # Import xfuser base components first
    from xfuser.core.distributed import (
        get_world_group, 
        get_runtime_state,
        get_data_parallel_rank,
        get_data_parallel_world_size,
        is_dp_last_group
    )
    from xfuser.config import FlexibleArgumentParser, EngineConfig
    
    # Now import the pipeline wrapper
    from xfuser import xFuserCogVideoXPipeline, xFuserArgs
except ImportError as e:
    print(f"Error importing xfuser components: {e}")
    print("Make sure xfuser package is installed correctly.")
    exit(1)

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="CogVideoX Generation with xFuser Distributed")
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
    parser.add_argument("--ulysses_degree", type=int, default=2, help="Ulysses attention degree")
    parser.add_argument("--ring_degree", type=int, default=2, help="Ring attention degree")
    
    # Distributed configuration
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed execution")
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    # Initialize distributed environment and get rank info
    args = parse_args()
    logger = setup_logging()
    
    # Ensure results directory exists (from rank 0 only)
    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        os.makedirs("results", exist_ok=True)
    
    # Create xfuser engine config using xfuser's argument parser
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
    
    # Build engine config from args
    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    
    # Get local rank from environment or args
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank
    
    # Set device and random seed
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed)
    
    # Load model with xfuser wrapper
    logger.info(f"Rank {local_rank}: Loading CogVideoX model from {args.model}")
    start_load_time = time.time()
    
    try:
        # Initialize the model with xfuser
        pipe = xFuserCogVideoXPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
        ).to(device)
        
        logger.info(f"Rank {local_rank}: Model loaded in {time.time() - start_load_time:.2f} seconds")
        
        # Apply optimizations if requested
        if args.enable_tiling:
            logger.info(f"Rank {local_rank}: Enabling VAE tiling")
            pipe.vae.enable_tiling()
        
        if args.enable_slicing:
            logger.info(f"Rank {local_rank}: Enabling VAE slicing")
            pipe.vae.enable_slicing()
        
        # Generate video
        logger.info(f"Rank {local_rank}: Starting video generation")
        logger.info(f"Rank {local_rank}: Settings - {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
        
        # Reset CUDA memory stats before main inference
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Create separate generator for each rank to ensure different outputs when needed
        # But keep the same base seed for consistency
        generator = torch.Generator(device=device).manual_seed(args.seed + local_rank)
        
        # Run the model
        output = pipe(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        
        # Collect performance metrics
        inference_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # Convert to GB
        
        logger.info(f"Rank {local_rank}: Generation completed in {inference_time:.2f} seconds")
        logger.info(f"Rank {local_rank}: Peak memory usage: {peak_memory:.2f} GB")
        
        # Save video output (only from the last data parallel group)
        if pipe.is_dp_last_group():
            # Create a descriptive filename with parallel configuration
            parallel_info = (
                f"dp{engine_config.parallel_config.dp_degree}_"
                f"cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{args.ulysses_degree}_ring{args.ring_degree}"
            )
            output_filename = f"results/cogvideox_{parallel_info}_{args.width}x{args.height}.mp4"
            
            # Save the video
            export_to_video(output.frames[0], output_filename, fps=8)
            logger.info(f"Rank {local_rank}: Video saved to {output_filename}")
        
        # Print final stats from rank 0
        if local_rank == 0:
            logger.info(f"\n=== Final Performance Summary ===")
            logger.info(f"Model: {args.model}")
            logger.info(f"Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")
            logger.info(f"Inference steps: {args.num_inference_steps}")
            logger.info(f"Total time: {inference_time:.2f} seconds")
            logger.info(f"Parallel config: ulysses={args.ulysses_degree}, ring={args.ring_degree}")
        
        # Clean up distributed environment
        get_runtime_state().destroy_distributed_env()
        
    except ImportError as e:
        logger.error(f"Rank {local_rank}: Import error: {e}")
        logger.error("Please ensure all dependencies are correctly installed")
    except RuntimeError as e:
        logger.error(f"Rank {local_rank}: Runtime error: {e}")
        logger.error("Check GPU memory and model compatibility")
    except Exception as e:
        logger.error(f"Rank {local_rank}: Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 