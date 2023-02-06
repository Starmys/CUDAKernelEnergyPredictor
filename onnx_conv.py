import os
import sys
import json

import torch
import numpy as np
import onnxruntime as ort

assert ort.get_device() == 'GPU'


RANDOM_SEED = 2022


def test_op(config):
    algo = config.pop('algo')
    gpu_id = config.pop('gpu_id')

    hw = config.pop('hw')
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn(1, config['in_channels'], hw, hw)
    padding = (config['kernel_size'] - 1) // 2  # get same padding
    conv_module = torch.nn.Conv2d(**config, padding=padding)
    torch.onnx.export(conv_module, input_tensor, f'tmp/conv_{gpu_id}.onnx')

    algo_preset_file = os.path.join(os.path.dirname(__file__), 'tmp', 'algo_preset.txt')
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

    ort_sess_warm = ort.InferenceSession(f'tmp/conv_{gpu_id}.onnx', providers=providers)
    for _ in range(200):  # Warm-Up
        output = ort_sess_warm.run(None, {'input': input_numpy})

    options = ort.SessionOptions()
    options.enable_profiling = True
    ort_sess_iter = ort.InferenceSession(f'tmp/conv_{gpu_id}.onnx', options, providers=providers)
    output = ort_sess_iter.run(None, {'input': input_numpy})

    prof_file = ort_sess_iter.end_profiling()
    with open(prof_file) as f:
        prof_results = json.loads(f.read())

    results = {}
    for record in prof_results:
        if record['cat'] == 'Node' and 'args' in record and 'gpu_latency' in record['args']:
            results = {
                'latency': eval(record['args']['gpu_latency']),
                'energy': eval(record['args']['gpu_energy'])[0],
            }
            break

    os.remove(prof_file)
    return results


if __name__ == '__main__':
    cfg = {}
    for arg in sys.argv[1:]:
        k, v = arg[2:].split('=')
        cfg[k] = int(v)
    print(json.dumps(test_op(cfg)))
