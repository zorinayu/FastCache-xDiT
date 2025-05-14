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

# Import pipeline implementations here to avoid circular imports
# These imports should be below the xFuserPipelineWrapperRegister class definition
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

# Now import the actual pipeline implementations for registration
try:
    # Import without causing circular imports
    from diffusers import DiffusionPipeline, HunyuanDiTPipeline, PixArtAlphaPipeline, PixArtSigmaPipeline
    from diffusers import StableDiffusion3Pipeline, FluxPipeline, LattePipeline, CogVideoXPipeline
    from diffusers import ConsisIDPipeline, StableDiffusionXLPipeline
    
    # Register specific pipeline wrappers
    if 'HunyuanDiTPipeline' in locals():
        xFuserPipelineWrapperRegister.register(HunyuanDiTPipeline, xFuserHunyuanDiTPipelineWrapper)
    
    if 'PixArtAlphaPipeline' in locals():
        xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline, xFuserPixArtAlphaPipelineWrapper)
    
    if 'PixArtSigmaPipeline' in locals():
        xFuserPipelineWrapperRegister.register(PixArtSigmaPipeline, xFuserPixArtSigmaPipelineWrapper)
    
    if 'StableDiffusion3Pipeline' in locals():
        xFuserPipelineWrapperRegister.register(StableDiffusion3Pipeline, xFuserStableDiffusion3PipelineWrapper)
    
    if 'FluxPipeline' in locals():
        xFuserPipelineWrapperRegister.register(FluxPipeline, xFuserFluxPipelineWrapper)
    
    if 'LattePipeline' in locals():
        xFuserPipelineWrapperRegister.register(LattePipeline, xFuserLattePipelineWrapper)
    
    if 'CogVideoXPipeline' in locals():
        xFuserPipelineWrapperRegister.register(CogVideoXPipeline, xFuserCogVideoXPipelineWrapper)
    
    if 'ConsisIDPipeline' in locals():
        xFuserPipelineWrapperRegister.register(ConsisIDPipeline, xFuserConsisIDPipelineWrapper)
    
    if 'StableDiffusionXLPipeline' in locals():
        xFuserPipelineWrapperRegister.register(StableDiffusionXLPipeline, xFuserStableDiffusionXLPipelineWrapper)

except ImportError as e:
    logger.warning(f"Some diffusers pipelines could not be imported: {e}")

# Register FastCache as a general wrapper
xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserFastCachePipelineWrapper)