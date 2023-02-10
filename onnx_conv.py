import os
import sys
import time
import json

import torch
import numpy as np
import onnxruntime as ort

from gpu_sampler import GPUSampler


assert ort.get_device() == 'GPU'


RANDOM_SEED = 2022
TMP_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp')


def test_op(config):
    algo = config.pop('algo')
    gpu_id = config.pop('gpu_id')
    use_onnx_profiler = config.pop('onnx_profiler') > 0
    profile_energy = config.pop('profile_energy') > 0
    num_warmups = config.pop('num_warmups')
    num_iters = 1 if use_onnx_profiler else config.pop('num_iters')

    hw = config.pop('hw')
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn(1, config['in_channels'], hw, hw)
    padding = (config['kernel_size'] - 1) // 2  # get same padding
    conv_module = torch.nn.Conv2d(**config, padding=padding)

    model_path = os.path.join(TMP_FOLDER, f'conv_{gpu_id}.onnx')
    torch.onnx.export(conv_module, input_tensor, model_path)

    algo_preset_file = os.path.join(TMP_FOLDER, 'algo_preset.txt')
    with open(algo_preset_file, 'w') as f:
        f.write(f'Conv_0 {algo}')
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': gpu_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'PRESET',
            'do_copy_in_default_stream': True,
            # 'algo_preset_file': algo_preset_file,
        }),
        'CPUExecutionProvider',
    ]
    input_numpy = input_tensor.numpy()

    ort_sess_warm = ort.InferenceSession(model_path, providers=providers)
    for _ in range(num_warmups):
        output = ort_sess_warm.run(None, {'input': input_numpy})

    options = ort.SessionOptions()
    options.enable_profiling = use_onnx_profiler
    ort_sess_iter = ort.InferenceSession(model_path, options, providers=providers)

    if not use_onnx_profiler:
        if profile_energy:
            x = GPUSampler(gpu_id=gpu_id)
            x.start()
        start = time.time()

    for _ in range(num_iters):
        output = ort_sess_iter.run(None, {'input': input_numpy})

    results = {}
    if use_onnx_profiler:
        prof_file = ort_sess_iter.end_profiling()
        with open(prof_file) as f:
            prof_results = json.loads(f.read())
        for record in prof_results:
            if record['cat'] == 'Node' and 'args' in record and 'gpu_latency' in record['args']:
                results['latency'] = eval(record['args']['gpu_latency'])
                if profile_energy:
                    results['energy'] = eval(record['args']['gpu_energy'])[0]
                break
        os.remove(prof_file)
    else:
        total_time = time.time() - start
        results['latency'] = total_time / num_iters
        if profile_energy:
            x.terminate()
            total_energy = x.calculate_energy()
            results['energy'] = total_energy / num_iters

    return results


if __name__ == '__main__':
    cfg = {}
    for arg in sys.argv[1:]:
        k, v = arg[2:].split('=')
        cfg[k] = int(v)
    print(json.dumps(test_op(cfg)))
