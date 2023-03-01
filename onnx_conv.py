import os
import sys
import time
import json

import torch
import onnxruntime as ort

from onnx_model_runner import OnnxModelRunner


assert ort.get_device() == 'GPU'


RANDOM_SEED = 2022
TMP_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp')


def test_op(config):
    algo = config['algo']
    gpu_id = config['gpu_id']
    num_warmups = config['num_warmups']
    num_iters = config['num_iters']

    workspace = os.path.join(TMP_FOLDER, f'profile-{gpu_id}')
    os.makedirs(workspace, exist_ok=True)

    hw = config['hw']
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn(1, config['in_channels'], hw, hw)
    padding = (config['kernel_size'] - 1) // 2  # get same padding
    conv_module = torch.nn.Conv2d(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        kernel_size=config['kernel_size'],
        stride=config['stride'],
        groups=config['groups'],
        padding=padding,
    )

    model_path = os.path.join(workspace, f'conv.onnx')
    torch.onnx.export(conv_module, input_tensor, model_path)

    algo_preset_path = os.path.join(workspace, 'algo_preset.txt')
    with open(algo_preset_path, 'w') as f:
        f.write(f'Conv_0 {algo}')

    results = OnnxModelRunner(workspace).run(
        model_path=model_path,
        algo_preset_path=algo_preset_path,
        warmup_repeat=num_warmups,
        num_repeat=num_iters,
        gpu_id=gpu_id,
    )
    return {'num_warmups': num_warmups, 'num_iters': num_iters, **results}


if __name__ == '__main__':
    cfg = {}
    for arg in sys.argv[1:]:
        k, v = arg[2:].split('=')
        cfg[k] = int(v)
    print(json.dumps(test_op(cfg)))
