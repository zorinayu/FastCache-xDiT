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


# Register basic diffusers pipeline
from diffusers import DiffusionPipeline

xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserPipelineBaseWrapper)

# Register Hunyuan DiT Pipeline
try:
    from diffusers import HunyuanDiTPipeline

    xFuserPipelineWrapperRegister.register(HunyuanDiTPipeline, xFuserHunyuanDiTPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import PixArtAlphaPipeline

    xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline, xFuserPixArtAlphaPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import PixArtSigmaPipeline

    xFuserPipelineWrapperRegister.register(PixArtSigmaPipeline, xFuserPixArtSigmaPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import StableDiffusion3Pipeline

    xFuserPipelineWrapperRegister.register(
        StableDiffusion3Pipeline, xFuserStableDiffusion3PipelineWrapper
    )
except ImportError:
    pass

try:
    from diffusers import FluxPipeline

    xFuserPipelineWrapperRegister.register(FluxPipeline, xFuserFluxPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import LattePipeline

    xFuserPipelineWrapperRegister.register(LattePipeline, xFuserLattePipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import CogVideoXPipeline

    xFuserPipelineWrapperRegister.register(CogVideoXPipeline, xFuserCogVideoXPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import ConsisIDPipeline
    xFuserPipelineWrapperRegister.register(ConsisIDPipeline, xFuserConsisIDPipelineWrapper)
except ImportError:
    pass

try:
    from diffusers import StableDiffusionXLPipeline
    xFuserPipelineWrapperRegister.register(StableDiffusionXLPipeline, xFuserStableDiffusionXLPipelineWrapper)
except ImportError:
    pass

# Register FastCache pipeline wrapper
# This can be used with any pipeline type as it's an enhancement wrapper
from diffusers import DiffusionPipeline
xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserFastCachePipelineWrapper)