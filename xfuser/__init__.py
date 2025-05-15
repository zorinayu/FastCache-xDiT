from xfuser.config import xFuserArgs, EngineConfig
from xfuser.parallel import xDiTParallel

# Define a lazy import mechanism to avoid circular imports
class LazyImport:
    def __init__(self, module_name, target_name):
        self.module_name = module_name
        self.target_name = target_name
        self._module = None

    def __call__(self, *args, **kwargs):
        if self._module is None:
            module = __import__(self.module_name, fromlist=[self.target_name])
            self._module = getattr(module, self.target_name)
        return self._module(*args, **kwargs)

    def __getattr__(self, name):
        if self._module is None:
            module = __import__(self.module_name, fromlist=[self.target_name])
            self._module = getattr(module, self.target_name)
        return getattr(self._module, name)

# Configure lazy imports for pipeline classes
xFuserPixArtAlphaPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_pixart_alpha", "xFuserPixArtAlphaPipeline")
xFuserPixArtSigmaPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_pixart_sigma", "xFuserPixArtSigmaPipeline")
xFuserStableDiffusion3Pipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_stable_diffusion_3", "xFuserStableDiffusion3Pipeline")
xFuserFluxPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_flux", "xFuserFluxPipeline")
xFuserLattePipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_latte", "xFuserLattePipeline")
xFuserHunyuanDiTPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_hunyuandit", "xFuserHunyuanDiTPipeline")
xFuserCogVideoXPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_cogvideox", "xFuserCogVideoXPipeline")
xFuserConsisIDPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_consisid", "xFuserConsisIDPipeline")
xFuserStableDiffusionXLPipeline = LazyImport("xfuser.model_executor.pipelines.pipeline_stable_diffusion_xl", "xFuserStableDiffusionXLPipeline")

__all__ = [
    "xFuserPixArtAlphaPipeline",
    "xFuserPixArtSigmaPipeline",
    "xFuserStableDiffusion3Pipeline",
    "xFuserFluxPipeline",
    "xFuserLattePipeline",
    "xFuserHunyuanDiTPipeline",
    "xFuserCogVideoXPipeline",
    "xFuserConsisIDPipeline",
    "xFuserStableDiffusionXLPipeline",
    "xFuserArgs",
    "EngineConfig",
    "xDiTParallel",
]
