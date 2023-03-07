import os
import math
import datetime
import subprocess

import pandas as pd

from onnx_conv import test_op


GPU_ID = 1  # 1 2 0 1 2 0
ALGO = 4    # 0 1 2 4 6 5
ALGO_ENUM = [
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT',
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING',
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD',
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED',
]
MODE = 'energy'

if MODE == 'energy':
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'search_space', 'regnet_convs_unique.csv')
    # CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'search_space', 'regnet_convs_expand.csv')
elif MODE == 'latency':
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'search_space', 'regnet_convs_expand.csv')
else:
    raise ValueError(f'mode not supported: {MODE}')

LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs', MODE)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
LOG_PATH = os.path.join(LOG_FOLDER, f'{ALGO}-{datetime.date.today().strftime("%Y%m%d")}.csv')

FEATURES = ['hw', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'groups']


def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    return stdout, stderr


def profile_energy(config):
    config['gpu_id'] = GPU_ID
    config['algo'] = ALGO
    config['profile_energy'] = 1
    config['num_warmups'] = 10
    config['num_iters'] = 100
    trial_results = test_op(config)
    if trial_results is None:
        return None
    config['num_iters'] = math.ceil(10 / trial_results['latency'])
    config['num_warmups'] = 200
    return test_op(config)


def profile_latency(config):
    config['gpu_id'] = GPU_ID
    config['algo'] = ALGO
    config['profile_energy'] = 0
    config['num_warmups'] = 200
    config['num_iters'] = 1000
    return test_op(config)


if __name__ == '__main__':
    first_record = True
    for i, row in pd.read_csv(CONFIG_PATH).iterrows():
        config = row.to_dict()
        print(f'#{i}: {config}')
        results = {
            'energy': profile_energy,
            'latency': profile_latency,
        }[MODE](config)
        if results is None:
            continue
        data = config | results
        if first_record:
            with open(LOG_PATH, 'w') as f:
                f.write(','.join(data.keys()) + '\n')
            first_record = False
        with open(LOG_PATH, 'a') as f:
            f.write(','.join([str(x) for x in data.values()]) + '\n')
