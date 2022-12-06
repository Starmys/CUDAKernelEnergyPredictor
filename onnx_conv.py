import sys
import time
import json

import torch
import numpy as np
import onnxruntime as ort

assert ort.get_device() == 'GPU'

from gpu_sampler import GPUSampler


RANDOM_SEED = 2022


def test_op(config):
    gpu_id = config.pop('gpu_id')
    sample_num = config.pop('sample_num')
    use_gpu_sampler = config.pop('gpu_sampler') > 0

    hw = config.pop('hw')
    torch.manual_seed(RANDOM_SEED)
    input = torch.randn(1, config['in_channels'], hw, hw)
    padding = (config['kernel_size'] - 1) // 2  # get same padding
    conv_module = torch.nn.Conv2d(**config, padding=padding)
    torch.onnx.export(conv_module, input, f'tmp/conv_{gpu_id}.onnx')

    ort_sess = ort.InferenceSession(f'tmp/conv_{gpu_id}.onnx', providers=[
        ('CUDAExecutionProvider', {
            'device_id': gpu_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'DEFAULT',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ])

    def call_func():
        outputs = ort_sess.run(None, {'input': input.numpy()})

    call_func()

    if use_gpu_sampler:
        x = GPUSampler(gpu_id=gpu_id)
        x.start()
    start = time.time()
    for _ in range(sample_num):
        call_func()
    total_time = time.time() - start
    if use_gpu_sampler:
        x.terminate()

    results = {}
    results['latency'] = total_time / sample_num

    if use_gpu_sampler:
        total_energy = x.calculate_energy()
        results['energy'] = total_energy / sample_num
        gpu_readings = x.export_readings()
        results['temperature'] = np.mean(gpu_readings['temperature'])
        results['memory'] = np.mean(gpu_readings['used_memory'])
        results['util'] = np.mean(gpu_readings['gpu_util'])

    return results


if __name__ == '__main__':
    cfg = {}
    for arg in sys.argv[1:]:
        k, v = arg[2:].split('=')
        cfg[k] = int(v)
    print(json.dumps(test_op(cfg)))
