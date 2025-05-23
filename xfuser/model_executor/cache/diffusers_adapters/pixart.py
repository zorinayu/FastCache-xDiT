"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import functools
import unittest.mock

import torch
from torch import nn
from diffusers import PixArtTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import register_transformer_adapter

from xfuser.model_executor.cache import utils
from xfuser.logger import init_logger

logger = init_logger(__name__)

def create_cached_transformer_blocks(use_cache, transformer, rel_l1_thresh, return_hidden_states_first, num_steps, motion_threshold=0.1):
    cached_transformer_class = {
        "Fb": utils.FBCachedTransformerBlocks,
        "Tea": utils.TeaCachedTransformerBlocks,
        "Fast": utils.FastCachedTransformerBlocks,
    }.get(use_cache)

    if not cached_transformer_class:
        raise ValueError(f"Unsupported use_cache value: {use_cache}")

    # For PixArt, single_transformer_blocks is None
    single_transformer_blocks = None
    
    # FastCache requires motion_threshold parameter
    if use_cache == "Fast":
        return cached_transformer_class(
            transformer.transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=return_hidden_states_first,
            num_steps=num_steps,
            motion_threshold=motion_threshold,
            name="pixart",
        )
    else:
        return cached_transformer_class(
            transformer.transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=return_hidden_states_first,
            num_steps=num_steps,
            name="pixart",
        )


def apply_cache_on_transformer(
    transformer: PixArtTransformer2DModel,
    *,
    rel_l1_thresh=0.05,
    return_hidden_states_first=True,
    num_steps=8,
    use_cache="Fast",
    motion_threshold=0.1,
):
    try:
        # Create cached transformer blocks
        cached_transformer_blocks = nn.ModuleList([
            create_cached_transformer_blocks(
                use_cache, 
                transformer, 
                rel_l1_thresh, 
                return_hidden_states_first, 
                num_steps,
                motion_threshold
            )
        ])

        # Store original forward function
        original_forward = transformer.forward

        # Create new forward function
        @functools.wraps(original_forward)
        def new_forward(
            self,
            *args,
            **kwargs,
        ):
            # Temporarily patch transformer_blocks with cached version
            with unittest.mock.patch.object(
                self,
                "transformer_blocks",
                cached_transformer_blocks,
            ):
                return original_forward(
                    *args,
                    **kwargs,
                )

        # Apply the new forward function
        transformer.forward = new_forward.__get__(transformer)

        logger.info(f"Applied {use_cache}Cache to PixArtTransformer2DModel")
    except Exception as e:
        logger.error(f"Error applying cache to PixArtTransformer2DModel: {e}")
    
    return transformer

# Register PixArt models
try:
    register_transformer_adapter(PixArtTransformer2DModel, "pixart")
    register_transformer_adapter(Transformer2DModel, "pixart")  # For compatibility with base class
    
    # Try to register the wrapper if it's available
    from xfuser.model_executor.models.transformers.pixart_transformer_2d import xFuserPixArtTransformer2DWrapper
    register_transformer_adapter(xFuserPixArtTransformer2DWrapper, "pixart")
except Exception as e:
    logger.warning(f"Could not register some PixArt adapters: {e}") 