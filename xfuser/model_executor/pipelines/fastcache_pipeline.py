import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from xfuser.logger import init_logger
from xfuser.model_executor.accelerator.fastcache import FastCacheAccelerator

logger = init_logger(__name__)


class xFuserFastCachePipelineWrapper:
    """
    Standalone FastCache pipeline wrapper for accelerated DiT inference
    This wrapper doesn't inherit from xFuser base classes to avoid distributed initialization
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.fastcache_enabled = False
        self.fastcache_accelerators = {}
        logger.info("Created standalone FastCache pipeline wrapper")
        
    def enable_fastcache(self, 
                         cache_ratio_threshold=0.05, 
                         motion_threshold=0.1, 
                         significance_level=0.05):
        """Enable FastCache acceleration on the pipeline's transformer"""
        logger.info(f"Enabling FastCache with threshold={cache_ratio_threshold}, motion={motion_threshold}")
        
        # Find and wrap transformer components
        if hasattr(self.pipeline, "transformer"):
            logger.info("Applying FastCache to transformer component")
            self._apply_fastcache_to_transformer(
                self.pipeline.transformer, 
                cache_ratio_threshold, 
                motion_threshold, 
                significance_level
            )
        elif hasattr(self.pipeline, "unet"):
            logger.info("Applying FastCache to unet component")
            self._apply_fastcache_to_unet(
                self.pipeline.unet, 
                cache_ratio_threshold, 
                motion_threshold, 
                significance_level
            )
        else:
            logger.warning("No transformer or unet found in pipeline")
            return
            
        self.fastcache_enabled = True
        logger.info("FastCache enabled successfully")
    
    def _apply_fastcache_to_transformer(self, transformer, cache_ratio_threshold, motion_threshold, significance_level):
        """Apply FastCache to transformer blocks individually"""
        wrapped_count = 0
        
        # Apply FastCache to transformer_blocks
        if hasattr(transformer, "transformer_blocks"):
            for i, block in enumerate(transformer.transformer_blocks):
                accelerator = FastCacheAccelerator(
                    model=block,
                    cache_ratio_threshold=cache_ratio_threshold,
                    motion_threshold=motion_threshold,
                    significance_level=significance_level
                )
                
                self.fastcache_accelerators[f"transformer_blocks.{i}"] = accelerator
                transformer.transformer_blocks[i] = accelerator
                wrapped_count += 1
        
        # Apply FastCache to single_transformer_blocks if they exist
        if hasattr(transformer, "single_transformer_blocks"):
            for i, block in enumerate(transformer.single_transformer_blocks):
                accelerator = FastCacheAccelerator(
                    model=block,
                    cache_ratio_threshold=cache_ratio_threshold,
                    motion_threshold=motion_threshold,
                    significance_level=significance_level
                )
                
                self.fastcache_accelerators[f"single_transformer_blocks.{i}"] = accelerator
                transformer.single_transformer_blocks[i] = accelerator
                wrapped_count += 1
        
        logger.info(f"Applied FastCache to {wrapped_count} transformer blocks")
    
    def _apply_fastcache_to_unet(self, unet, cache_ratio_threshold, motion_threshold, significance_level):
        """Apply FastCache to UNet transformer blocks"""
        def wrap_transformer_blocks(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if hasattr(child, "transformer_blocks"):
                    # Wrap individual blocks in this module
                    for i, block in enumerate(child.transformer_blocks):
                        accelerator = FastCacheAccelerator(
                            model=block,
                            cache_ratio_threshold=cache_ratio_threshold,
                            motion_threshold=motion_threshold,
                            significance_level=significance_level
                        )
                        
                        self.fastcache_accelerators[f"{full_name}.transformer_blocks.{i}"] = accelerator
                        child.transformer_blocks[i] = accelerator
                    
                    logger.info(f"Applied FastCache to transformer blocks in {full_name}")
                else:
                    # Recursively check children
                    wrap_transformer_blocks(child, full_name)
        
        wrap_transformer_blocks(unet)
    
    def get_cache_statistics(self):
        """Get cache hit statistics from all accelerators"""
        stats = {}
        for name, accelerator in self.fastcache_accelerators.items():
            stats[name] = {
                "cache_hits": int(accelerator.cache_hits.item()) if hasattr(accelerator, "cache_hits") else 0,
                "total_steps": int(accelerator.total_steps.item()) if hasattr(accelerator, "total_steps") else 0,
                "cache_hit_ratio": float(accelerator.cache_hits / accelerator.total_steps) if hasattr(accelerator, "total_steps") and accelerator.total_steps > 0 else 0.0
            }
        return stats
    
    def reset_cache_statistics(self):
        """Reset cache statistics for all accelerators"""
        for accelerator in self.fastcache_accelerators.values():
            if hasattr(accelerator, "cache_hits"):
                accelerator.cache_hits.zero_()
            if hasattr(accelerator, "total_steps"):
                accelerator.total_steps.zero_()
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped pipeline with FastCache acceleration"""
        if not self.fastcache_enabled:
            logger.warning("FastCache not enabled, calling original pipeline")
        
        try:
            return self.pipeline(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped pipeline"""
        if name in ['pipeline', 'fastcache_enabled', 'fastcache_accelerators']:
            return object.__getattribute__(self, name)
        
        try:
            return getattr(self.pipeline, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") 