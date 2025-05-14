from typing import Dict, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.logger import init_logger
from .base_pipeline import xFuserPipelineBaseWrapper
from .pipeline_hunyuandit import xFuserHunyuanDiTPipeline
from .pipeline_pixart_alpha import xFuserPixArtAlphaPipeline
from .pipeline_pixart_sigma import xFuserPixArtSigmaPipeline
from .pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline
from .pipeline_flux import xFuserFluxPipeline
from .pipeline_latte import xFuserLattePipeline
from .pipeline_cogvideox import xFuserCogVideoXPipeline
from .pipeline_consisid import xFuserConsisIDPipeline
from .pipeline_stable_diffusion_xl import xFuserStableDiffusionXLPipeline
from .fastcache_pipeline import xFuserFastCachePipelineWrapper
from .dit_pipeline import (
    xFuserHunyuanDiTPipelineWrapper,
    xFuserPixArtAlphaPipelineWrapper,
    xFuserPixArtSigmaPipelineWrapper,
    xFuserStableDiffusion3PipelineWrapper,
    xFuserFluxPipelineWrapper,
    xFuserLattePipelineWrapper,
    xFuserCogVideoXPipelineWrapper,
    xFuserConsisIDPipelineWrapper,
    xFuserStableDiffusionXLPipelineWrapper,
)

logger = init_logger(__name__)

class xFuserPipelineWrapperRegister:
    _registry = {}

    @classmethod
    def register(cls, pipeline_class, wrapper_class):
        cls._registry[pipeline_class] = wrapper_class

    @classmethod
    def get_class(cls, pipeline):
        """
        Get the wrapper class of the pipeline class.
        If not found, try to get the pipeline class's base class.
        If still not found, use the default wrapper class.
        """
        pipeline_class = pipeline.__class__
        if pipeline_class in cls._registry:
            return cls._registry[pipeline_class]
        else:
            for base in pipeline_class.__bases__:
                if base in cls._registry:
                    return cls._registry[base]
        return xFuserPipelineBaseWrapper

# Register base pipeline wrapper
xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserPipelineBaseWrapper)

# Don't directly import wrapper classes here to avoid circular imports
# The registration will happen when those modules are imported elsewhere

# Register FastCache wrapper directly since it doesn't have circular dependencies
from .fastcache_pipeline import xFuserFastCachePipelineWrapper
xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserFastCachePipelineWrapper)

# Dynamically register pipelines to avoid circular imports
def register_all_pipelines():
    """Register all pipeline wrappers after modules are fully loaded."""
    try:
        # Import diffusers classes
        from diffusers import (DiffusionPipeline, HunyuanDiTPipeline, PixArtAlphaPipeline, 
                              PixArtSigmaPipeline, StableDiffusion3Pipeline, 
                              FluxPipeline, LattePipeline, CogVideoXPipeline,
                              ConsisIDPipeline, StableDiffusionXLPipeline)
        
        # Import wrapper classes
        from .dit_pipeline import (
            xFuserHunyuanDiTPipelineWrapper,
            xFuserPixArtAlphaPipelineWrapper,
            xFuserPixArtSigmaPipelineWrapper,
            xFuserStableDiffusion3PipelineWrapper,
            xFuserFluxPipelineWrapper,
            xFuserLattePipelineWrapper,
            xFuserCogVideoXPipelineWrapper,
            xFuserConsisIDPipelineWrapper,
            xFuserStableDiffusionXLPipelineWrapper,
        )
        
        # Register all pipeline wrapper mappings
        pipeline_mappings = {
            HunyuanDiTPipeline: xFuserHunyuanDiTPipelineWrapper,
            PixArtAlphaPipeline: xFuserPixArtAlphaPipelineWrapper,
            PixArtSigmaPipeline: xFuserPixArtSigmaPipelineWrapper,
            StableDiffusion3Pipeline: xFuserStableDiffusion3PipelineWrapper,
            FluxPipeline: xFuserFluxPipelineWrapper,
            LattePipeline: xFuserLattePipelineWrapper,
            CogVideoXPipeline: xFuserCogVideoXPipelineWrapper,
            ConsisIDPipeline: xFuserConsisIDPipelineWrapper,
            StableDiffusionXLPipeline: xFuserStableDiffusionXLPipelineWrapper,
        }
        
        # Register each mapping if the diffusers class is available
        for pipeline_class, wrapper_class in pipeline_mappings.items():
            if pipeline_class.__name__ in globals():
                xFuserPipelineWrapperRegister.register(pipeline_class, wrapper_class)
                logger.debug(f"Registered {pipeline_class.__name__} with {wrapper_class.__name__}")
    
    except ImportError as e:
        logger.warning(f"Some diffusers pipelines could not be imported: {e}")
    except Exception as e:
        logger.error(f"Error registering pipelines: {e}")

# This will be called after all modules are imported
# The actual registration will happen when needed
# register_all_pipelines()