"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import dataclasses
from typing import Dict, Optional, List
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch
from torch.nn import Module
from abc import ABC, abstractmethod
import math


# --------- CacheContext --------- #
class CacheContext(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("default_coef", torch.tensor([1.0, 0.0]).cuda())
        self.register_buffer("flux_coef", torch.tensor([498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]).cuda())
        
        self.register_buffer("original_hidden_states", None, persistent=False)
        self.register_buffer("original_encoder_hidden_states", None, persistent=False)
        self.register_buffer("hidden_states_residual", None, persistent=False)
        self.register_buffer("encoder_hidden_states_residual", None, persistent=False)
        self.register_buffer("modulated_inputs", None, persistent=False)
        
        # For FastCache
        self.register_buffer("prev_hidden_states", None, persistent=False)
        self.register_buffer("static_token_mask", None, persistent=False)
        
    def get_coef(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_coef")

#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: Optional[torch.nn.Module] = None
    transformer_blocks: Optional[List[torch.nn.Module]] = None
    single_transformer_blocks: Optional[List[torch.nn.Module]] = None
    cache_context: Optional[CacheContext] = None
    rel_l1_thresh: float = 0.6
    return_hidden_states_first: bool = True
    use_cache: torch.Tensor = torch.tensor(False, dtype=torch.bool)
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs): pass
    def on_forward_begin(self, state: CacheState, **kwargs): pass
    def on_forward_remaining_begin(self, state: CacheState, **kwargs): pass
    def on_forward_end(self, state: CacheState, **kwargs): pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks: Optional[List[CacheCallback]] = None):
        self.callbacks = list(callbacks) if callbacks else []

    def trigger_event(self, event: str, state: CacheState):
        for cb in self.callbacks:
            getattr(cb, event)(state)

# --------- Vectorized Poly1D --------- #
class VectorizedPoly1D(Module):
    def __init__(self, coefficients: torch.Tensor):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.degree = len(coefficients) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        return_hidden_states_first: bool = True,
        num_steps: int = -1,
        name: str = "default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = torch.nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.register_buffer("cnt", torch.tensor(0).cuda())
        self.register_buffer("accumulated_rel_l1_distance", torch.tensor([0.0]).cuda())
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool).cuda())

        self.cache_context = CacheContext()
        self.callback_handler = CallbackHandler(callbacks)

        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.return_hidden_states_first = return_hidden_states_first
        self.num_steps = num_steps
        self.name = name
        self.callback_handler.trigger_event("on_init_begin", self)

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()

    @abstractmethod
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor: pass

    @abstractmethod
    def get_start_idx(self) -> int: pass

    @abstractmethod
    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs): pass

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        if self.single_transformer_blocks:
            hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)

        self.cache_context.hidden_states_residual = hidden - self.cache_context.original_hidden_states
        self.cache_context.encoder_hidden_states_residual = encoder - self.cache_context.original_encoder_hidden_states
        return hidden, encoder

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        modulated, prev_modulated, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh) \
            if prev_modulated is not None else torch.tensor(False, dtype=torch.bool)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual
        else:
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)

        self.callback_handler.trigger_event("on_forward_end", self)
        return ((hidden, encoder) if self.return_hidden_states_first else (encoder, hidden))


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)

    def get_start_idx(self) -> int:
        return 1

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return self.l1_distance(t1, t2) < threshold

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        hidden_states, encoder_hidden_states = (hidden_states, encoder_hidden_states) if self.return_hidden_states_first else (encoder_hidden_states, hidden_states)
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = self.cache_context.modulated_inputs
        if not self.use_cache:
           self.cache_context.modulated_inputs = first_hidden_states_residual

        return first_hidden_states_residual, prev_first_hidden_states_residual, hidden_states, encoder_hidden_states


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        self.rescale_func = VectorizedPoly1D(self.cache_context.get_coef(self.name))

    def get_start_idx(self) -> int:
        return 0

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        diff = self.l1_distance(t1, t2)
        new_accum = self.accumulated_rel_l1_distance + self.rescale_func(diff)
        reset_mask = (self.cnt == 0) or (self.cnt == self.num_steps - 1)
        self.use_cache = torch.logical_and(new_accum < threshold, torch.logical_not(reset_mask))
        self.accumulated_rel_l1_distance[0] = torch.where(self.use_cache, new_accum[0], 0.0)
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)

        return self.use_cache

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        inp = hidden_states.clone()
        temb_ = kwargs.get("temb", None).clone()
        modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        prev_modulated = self.cache_context.modulated_inputs
        self.cache_context.modulated_inputs = modulated
        return modulated, prev_modulated, hidden_states, encoder_hidden_states


class FastCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.05,  # Default for FastCache is lower
        motion_threshold=0.1,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        
        # FastCache specific parameters
        self.motion_threshold = motion_threshold
        self.register_buffer("cache_hits", torch.tensor(0).cuda())
        self.register_buffer("total_steps", torch.tensor(0).cuda())
        
        # Initialize cache adaptation parameters
        self.beta0 = 0.01
        self.beta1 = 0.5
        self.beta2 = -0.002
        self.beta3 = 0.00005
        
        # Linear approximation for static tokens
        if hasattr(transformer_blocks[0], "config"):
            hidden_size = transformer_blocks[0].config.hidden_size
        else:
            # Estimate hidden size from the first block
            hidden_size = next(transformer_blocks[0].parameters()).shape[-1]
            
        self.cache_projection = torch.nn.Linear(hidden_size, hidden_size).cuda()

    def get_start_idx(self) -> int:
        return 0  # Process all blocks when not caching
    
    def get_adaptive_threshold(self, variance_score, timestep=None):
        """Calculate adaptive threshold based on variance and current timestep"""
        if timestep is None:
            timestep = self.cnt
            
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range
        return (self.beta0 + 
                self.beta1 * variance_score + 
                self.beta2 * normalized_timestep + 
                self.beta3 * normalized_timestep**2)

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        """Using FastCache's relative change metric for caching decision"""
        if t1 is None or t2 is None:
            return torch.tensor(False, dtype=torch.bool).cuda()
            
        # Compute relative change (Frobenius norm)
        delta = self.l1_distance(t1, t2)
        
        # Update total steps counter
        self.total_steps += 1
        
        # Compute statistical threshold based on chi-square
        n, d = t1.shape[0], t1.shape[1]  # token count, hidden dim
        dof = n * d  # degrees of freedom
        
        # Approximate chi-square threshold using normal distribution for large DOF
        z = 1.96  # z-score for 95% confidence
        chi2_threshold = dof + z * math.sqrt(2 * dof)
        statistical_threshold = math.sqrt(chi2_threshold / dof)
        
        # Adaptive threshold based on variance and timestep
        adaptive_threshold = self.get_adaptive_threshold(delta, self.cnt)
        
        # Final threshold combines statistical validity with adaptive behavior
        final_threshold = max(threshold, min(statistical_threshold, adaptive_threshold))
        
        # Cache decision
        use_cache = delta <= final_threshold
        
        # Update cache hits counter
        self.cache_hits += use_cache.int()
        
        return use_cache

    def compute_motion_saliency(self, hidden_states):
        """Compute motion saliency for spatial token reduction"""
        if self.cache_context.prev_hidden_states is None:
            return torch.ones(hidden_states.shape[1], device=hidden_states.device)
            
        # Compute token-wise differences
        token_diffs = (hidden_states - self.cache_context.prev_hidden_states).abs()
        
        # Take max across feature dimension to get token saliency
        token_saliency = token_diffs.max(dim=-1)[0].squeeze(0)
        
        # Normalize saliency
        if token_saliency.max() > 0:
            token_saliency = token_saliency / token_saliency.max()
            
        return token_saliency

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Store current hidden states for later comparisons
        prev_hidden_states = self.cache_context.prev_hidden_states
        
        # First run: just store hidden states and process normally
        if prev_hidden_states is None:
            self.cache_context.prev_hidden_states = hidden_states.detach().clone()
            return hidden_states, None, hidden_states, encoder_hidden_states
        
        # Compute motion saliency and mask for token reduction
        motion_saliency = self.compute_motion_saliency(hidden_states)
        self.cache_context.static_token_mask = motion_saliency <= self.motion_threshold
        
        # Update cached states
        self.cache_context.prev_hidden_states = hidden_states.detach().clone()
        
        return hidden_states, prev_hidden_states, hidden_states, encoder_hidden_states
    
    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """Override to implement spatial token reduction for FastCache"""
        # If using cache, shortcut and return
        if self.use_cache:
            return hidden, encoder
        
        # Check if we can use spatial token reduction
        static_mask = self.cache_context.static_token_mask
        if static_mask is not None and static_mask.any() and not static_mask.all():
            batch_size, seq_len, hidden_dim = hidden.shape
            
            # Split tokens into motion and static
            motion_indices = torch.where(~static_mask)[0]
            static_indices = torch.where(static_mask)[0]
            
            if len(motion_indices) > 0:
                # Get motion tokens
                motion_hidden = hidden.index_select(1, motion_indices)
                motion_encoder = encoder.index_select(1, motion_indices) if encoder is not None else None
                
                # Process motion tokens through transformer blocks
                processed_motion_hidden, processed_motion_encoder = super().process_blocks(
                    start_idx, motion_hidden, motion_encoder, *args, **kwargs
                )
                
                # Process static tokens through linear projection
                static_hidden = hidden.index_select(1, static_indices)
                static_encoder = encoder.index_select(1, static_indices) if encoder is not None else None
                static_hidden = self.cache_projection(static_hidden)
                
                # Combine results
                result_hidden = hidden.clone()
                result_hidden.index_copy_(1, motion_indices, processed_motion_hidden)
                result_hidden.index_copy_(1, static_indices, static_hidden)
                
                result_encoder = encoder.clone() if encoder is not None else None
                if result_encoder is not None:
                    result_encoder.index_copy_(1, motion_indices, processed_motion_encoder)
                    result_encoder.index_copy_(1, static_indices, static_encoder)
                
                return result_hidden, result_encoder
        
        # Fall back to normal processing
        return super().process_blocks(start_idx, hidden, encoder, *args, **kwargs)