import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.logger import init_logger
from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper

logger = init_logger(__name__)


class xFuserFastCachePipelineWrapper(xFuserPipelineBaseWrapper):
    """
    FastCache pipeline wrapper for accelerated DiT inference
    """
    
    def __init__(self, pipeline, engine_config=None):
        # FastCache wrapper doesn't need an engine_config, we'll create a minimal one
        if engine_config is None:
            from xfuser.config import (
                EngineConfig, ModelConfig, RuntimeConfig, ParallelConfig,
                DataParallelConfig, SequenceParallelConfig, TensorParallelConfig, PipeFusionParallelConfig
            )
            from xfuser.config.config import FastAttnConfig
            
            # Create minimal configs with required parameters
            model_config = ModelConfig(model="dummy")
            runtime_config = RuntimeConfig()
            
            # Create parallel config with all required sub-configs
            dp_config = DataParallelConfig(dit_parallel_size=1)
            sp_config = SequenceParallelConfig(dit_parallel_size=1)
            tp_config = TensorParallelConfig(dit_parallel_size=1)
            pp_config = PipeFusionParallelConfig(dit_parallel_size=1)
            
            parallel_config = ParallelConfig(
                dp_config=dp_config,
                sp_config=sp_config,
                tp_config=tp_config,
                pp_config=pp_config,
                world_size=1,
                dit_parallel_size=1,
                vae_parallel_size=0
            )
            
            fast_attn_config = FastAttnConfig()
            
            engine_config = EngineConfig(
                model_config=model_config,
                runtime_config=runtime_config,
                parallel_config=parallel_config,
                fast_attn_config=fast_attn_config
            )
        
        super().__init__(pipeline, engine_config)
        self.fastcache_enabled = False
        self.fastcache_accelerators = {}
        
    def enable_fastcache(self, 
                         cache_ratio_threshold=0.05, 
                         motion_threshold=0.1, 
                         significance_level=0.05):
        """Enable FastCache acceleration for this pipeline"""
        self.fastcache_enabled = True
        
        # Apply FastCache to transformer blocks in the model
        # Check different possible model architectures
        if hasattr(self.module, "transformer"):
            logger.info("Applying FastCache to Transformer model")
            self._apply_fastcache_to_transformer(
                self.module.transformer,
                cache_ratio_threshold,
                motion_threshold,
                significance_level
            )
        elif hasattr(self.module, "unet"):
            logger.info("Applying FastCache to UNet model")
            self._apply_fastcache_to_unet(
                self.module.unet,
                cache_ratio_threshold,
                motion_threshold,
                significance_level
            )
        else:
            logger.warning("Could not identify transformer blocks in the model for FastCache")
        
        return self
    
    def _apply_fastcache_to_transformer(self, transformer, cache_ratio_threshold, motion_threshold, significance_level):
        """Apply FastCache to transformer blocks in DiT models"""
        # For DiT models, we usually have transformer_blocks
        wrapped_blocks = 0
        
        if hasattr(transformer, "transformer_blocks"):
            for i, block in enumerate(transformer.transformer_blocks):
                accelerator = FastCacheAccelerator(
                    block,
                    cache_ratio_threshold=cache_ratio_threshold,
                    motion_threshold=motion_threshold,
                    significance_level=significance_level
                )
                
                self.fastcache_accelerators[f"transformer_blocks.{i}"] = accelerator
                transformer.transformer_blocks[i] = accelerator
                wrapped_blocks += 1
                logger.info(f"Applied FastCache to transformer block {i}")
        
        # Also check for single_transformer_blocks (like in Flux)
        if hasattr(transformer, "single_transformer_blocks"):
            for i, block in enumerate(transformer.single_transformer_blocks):
                accelerator = FastCacheAccelerator(
                    block,
                    cache_ratio_threshold=cache_ratio_threshold,
                    motion_threshold=motion_threshold,
                    significance_level=significance_level
                )
                
                self.fastcache_accelerators[f"single_transformer_blocks.{i}"] = accelerator
                transformer.single_transformer_blocks[i] = accelerator
                wrapped_blocks += 1
                logger.info(f"Applied FastCache to single transformer block {i}")
        
        logger.info(f"Applied FastCache to {wrapped_blocks} transformer blocks in the model")
    
    def _apply_fastcache_to_unet(self, unet, cache_ratio_threshold, motion_threshold, significance_level):
        """Apply FastCache to transformer blocks in UNet models"""
        # This method handles the specific UNet architecture used in diffusion models
        
        block_types_to_wrap = [
            "BasicTransformerBlock",  # Common for DiT models
            "Transformer2DModel",     # Found in some UNet implementations
            "SpatialTransformer"      # Used in StableDiffusion UNet
        ]
        
        # Keep track of wrapped blocks
        wrapped_blocks = 0
        
        # Helper function to recursively find and wrap transformer blocks
        def wrap_transformer_blocks(module, prefix=""):
            nonlocal wrapped_blocks
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if the module is a transformer block
                module_type = child.__class__.__name__
                if any(block_type in module_type for block_type in block_types_to_wrap):
                    # Create FastCache accelerator for this block
                    accelerator = FastCacheAccelerator(
                        child,
                        cache_ratio_threshold=cache_ratio_threshold,
                        motion_threshold=motion_threshold,
                        significance_level=significance_level
                    )
                    
                    # Store accelerator for later use
                    self.fastcache_accelerators[full_name] = accelerator
                    
                    # Replace the original module with the accelerator
                    setattr(module, name, accelerator)
                    wrapped_blocks += 1
                    
                    logger.info(f"Applied FastCache to {module_type} at {full_name}")
                else:
                    # Recursively process child modules
                    wrap_transformer_blocks(child, full_name)
        
        # Start the recursive wrapping
        wrap_transformer_blocks(unet)
        logger.info(f"Applied FastCache to {wrapped_blocks} transformer blocks in the model")
    
    def prepare_run(self, input_config):
        """Prepare the pipeline for running"""
        # FastCache doesn't need special preparation
        pass
    
    def get_cache_statistics(self):
        """Get cache hit statistics for all accelerators"""
        stats = {}
        for name, accelerator in self.fastcache_accelerators.items():
            stats[name] = {
                "cache_hit_ratio": accelerator.get_cache_hit_ratio(),
                "layer_stats": accelerator.get_layer_hit_stats()
            }
        return stats
    
    def reset_cache_statistics(self):
        """Reset cache statistics for all accelerators"""
        for accelerator in self.fastcache_accelerators.values():
            accelerator.reset_stats()
    
    def __call__(self, *args, **kwargs):
        """Main call method required by the abstract base class"""
        
        # If FastCache is not enabled, fall back to the original pipeline
        if not self.fastcache_enabled or not self.fastcache_accelerators:
            return self.module(*args, **kwargs)
        
        # Process with FastCache-wrapped pipeline
        result = self.module(*args, **kwargs)
        
        # Log cache statistics
        total_hits = 0
        total_steps = 0
        for name, accelerator in self.fastcache_accelerators.items():
            hits = accelerator.cache_hits
            steps = accelerator.total_steps
            total_hits += hits
            total_steps += steps
            if steps > 0:
                logger.debug(f"FastCache {name}: {hits}/{steps} hits ({hits/steps:.2%})")
        
        if total_steps > 0:
            logger.info(f"FastCache overall: {total_hits}/{total_steps} hits ({total_hits/total_steps:.2%})")
        
        return result
    
    def forward(self, *args, **kwargs):
        """Forward pass for the pipeline, with FastCache if enabled"""
        return self.__call__(*args, **kwargs) 