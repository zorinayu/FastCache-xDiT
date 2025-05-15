from xfuser.config import xFuserArgs, EngineConfig
from xfuser.parallel import xDiTParallel

# 使用函数导入模式来避免循环导入
def _import_pipelines():
    """延迟导入所有pipeline类，避免循环导入问题"""
    from xfuser.model_executor.pipelines import (
        xFuserPixArtAlphaPipeline,
        xFuserPixArtSigmaPipeline,
        xFuserStableDiffusion3Pipeline,
        xFuserFluxPipeline,
        xFuserLattePipeline,
        xFuserHunyuanDiTPipeline,
        xFuserCogVideoXPipeline,
        xFuserConsisIDPipeline,
        xFuserStableDiffusionXLPipeline
    )
    
    # 返回所有导入的类供外部使用
    return {
        "xFuserPixArtAlphaPipeline": xFuserPixArtAlphaPipeline,
        "xFuserPixArtSigmaPipeline": xFuserPixArtSigmaPipeline,
        "xFuserStableDiffusion3Pipeline": xFuserStableDiffusion3Pipeline,
        "xFuserFluxPipeline": xFuserFluxPipeline,
        "xFuserLattePipeline": xFuserLattePipeline,
        "xFuserHunyuanDiTPipeline": xFuserHunyuanDiTPipeline,
        "xFuserCogVideoXPipeline": xFuserCogVideoXPipeline,
        "xFuserConsisIDPipeline": xFuserConsisIDPipeline,
        "xFuserStableDiffusionXLPipeline": xFuserStableDiffusionXLPipeline,
    }

# 创建一个用来统一管理延迟导入的类
class LazyPipelinesImporter:
    def __init__(self):
        self._modules = None
        
    def __getattr__(self, name):
        # 首次访问任何属性时，才执行导入
        if self._modules is None:
            self._modules = _import_pipelines()
        
        if name in self._modules:
            return self._modules[name]
        
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 创建延迟导入对象
_lazy_pipelines = LazyPipelinesImporter()

# 为了向后兼容，在模块级别导出pipeline类
xFuserPixArtAlphaPipeline = property(lambda _: _lazy_pipelines.xFuserPixArtAlphaPipeline)
xFuserPixArtSigmaPipeline = property(lambda _: _lazy_pipelines.xFuserPixArtSigmaPipeline)
xFuserStableDiffusion3Pipeline = property(lambda _: _lazy_pipelines.xFuserStableDiffusion3Pipeline)
xFuserFluxPipeline = property(lambda _: _lazy_pipelines.xFuserFluxPipeline)
xFuserLattePipeline = property(lambda _: _lazy_pipelines.xFuserLattePipeline)
xFuserHunyuanDiTPipeline = property(lambda _: _lazy_pipelines.xFuserHunyuanDiTPipeline)
xFuserCogVideoXPipeline = property(lambda _: _lazy_pipelines.xFuserCogVideoXPipeline)
xFuserConsisIDPipeline = property(lambda _: _lazy_pipelines.xFuserConsisIDPipeline)
xFuserStableDiffusionXLPipeline = property(lambda _: _lazy_pipelines.xFuserStableDiffusionXLPipeline)

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
