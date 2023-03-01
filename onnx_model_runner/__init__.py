import os
from . import onnx_model_runner_backend as runner

class OnnxModelRunner(object):
    def __init__(
        self,
        workspace: str,
        save_optimized_model: bool = False,
        save_gpu_readings: bool = False,
        save_running_logs: bool = False,
        enable_profiling: bool = False,
    ):
        self._workspace = workspace
        os.makedirs(self._workspace, exist_ok=True)
        self._extra_funcs = {}
        if save_optimized_model:
            self._extra_funcs['optimized_model'] = 'onnx'
        if save_gpu_readings:
            self._extra_funcs['gpu_readings'] = 'csv'
        if save_running_logs:
            self._extra_funcs['running_logs'] = 'txt'
        if enable_profiling:
            self._extra_funcs['profile'] = 'json'
        for name, suffix in self._extra_funcs.items():
            os.makedirs(os.path.join(workspace, name), exist_ok=True)
    
    def run(
        self,
        model_path: str,
        algo_preset_path: str,
        num_repeat: int = 1,
        warmup_repeat: int = 0,
        time_limit : float = 10.0,
        gpu_id: int = 0,
        gpu_sampling_interval: float = 0.01,
    ):
        assert os.path.exists(model_path)
        assert os.path.isfile(model_path)
        cfg = runner.ONNXRunConfig()
        cfg.num_repeat = num_repeat
        cfg.warm_up_repeat = warmup_repeat
        cfg.time_limit = time_limit
        cfg.gpu_id = gpu_id
        cfg.gpu_sampling_interval = gpu_sampling_interval
        base_name = os.path.basename(model_path)
        for name, suffix in self._extra_funcs.items():
            save_path = os.path.join(self._workspace, name, f'{base_name}.{suffix}')
            setattr(cfg, f'{name}_save_path', save_path)
        res = runner.run_onnx_model(model_path, cfg, algo_preset_path)
        return {'latency': res[0], 'energy': res[1]}
