from .base_pipeline import xFuserPipelineBaseWrapper

# 使用延迟导入机制，避免在模块加载时进行过早的导入
class LazyLoader:
    """延迟加载器，仅在首次访问时才导入对应的模块"""
    
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._name = name
        self._module = None
    
    def _load(self):
        """导入模块，并缓存结果"""
        if self._module is None:
            # 执行实际导入
            module_path = f".{self._name}"
            module = __import__(self._name, globals=self._parent_module_globals, 
                               fromlist=[self._local_name], level=1)
            
            # 获取指定的模块属性
            module_attr = getattr(module, self._local_name)
            
            # 缓存
            self._module = module_attr
            
            # 将导入的属性替换到父模块的globals中
            self._parent_module_globals[self._local_name] = module_attr
        
        return self._module
    
    def __getattr__(self, attr):
        # 任何属性访问时加载模块
        module = self._load()
        return getattr(module, attr)
    
    def __call__(self, *args, **kwargs):
        # 作为函数调用时加载模块
        module = self._load()
        return module(*args, **kwargs)
    
# 使用延迟加载器定义所有的导入
xFuserPixArtAlphaPipeline = LazyLoader("xFuserPixArtAlphaPipeline", globals(), "pipeline_pixart_alpha")
xFuserPixArtSigmaPipeline = LazyLoader("xFuserPixArtSigmaPipeline", globals(), "pipeline_pixart_sigma")
xFuserStableDiffusion3Pipeline = LazyLoader("xFuserStableDiffusion3Pipeline", globals(), "pipeline_stable_diffusion_3")
xFuserFluxPipeline = LazyLoader("xFuserFluxPipeline", globals(), "pipeline_flux")
xFuserLattePipeline = LazyLoader("xFuserLattePipeline", globals(), "pipeline_latte")
xFuserCogVideoXPipeline = LazyLoader("xFuserCogVideoXPipeline", globals(), "pipeline_cogvideox")
xFuserConsisIDPipeline = LazyLoader("xFuserConsisIDPipeline", globals(), "pipeline_consisid")
xFuserHunyuanDiTPipeline = LazyLoader("xFuserHunyuanDiTPipeline", globals(), "pipeline_hunyuandit")
xFuserStableDiffusionXLPipeline = LazyLoader("xFuserStableDiffusionXLPipeline", globals(), "pipeline_stable_diffusion_xl")

__all__ = [
    "xFuserPipelineBaseWrapper",
    "xFuserPixArtAlphaPipeline",
    "xFuserPixArtSigmaPipeline",
    "xFuserStableDiffusion3Pipeline",
    "xFuserFluxPipeline",
    "xFuserLattePipeline",
    "xFuserHunyuanDiTPipeline",
    "xFuserCogVideoXPipeline",
    "xFuserConsisIDPipeline",
    "xFuserStableDiffusionXLPipeline",
]