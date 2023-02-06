import os
import math
import json
import datetime
import subprocess

import pandas as pd


GPU_ID = 4  # 4 5 6 7 7
ALGO = 0  # 0 1 2 4 6
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
    CONFIG_PATH = os.path.join('search_space', 'regnet_convs_unique.csv')
    # CONFIG_PATH = os.path.join('search_space', 'regnet_convs_expand.csv')
elif MODE == 'latency':
    CONFIG_PATH = os.path.join('search_space', 'regnet_convs_expand.csv')
elif MODE == 'ncu':
    CONFIG_PATH = os.path.join('search_space', 'regnet_convs_unique.csv')
else:
    raise ValueError(f'mode not supported: {MODE}')

LOG_FOLDER = os.path.join('logs', MODE)
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
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    stdout, stderr = run_cmd(python_cmd)
    if len(stderr) > 0:
        print(stderr)
        return None
    return json.loads(stdout.split('\n')[-2])


def profile_latency(config):
    config['gpu_id'] = GPU_ID
    config['algo'] = ALGO
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    stdout, stderr = run_cmd(python_cmd)
    if len(stderr) > 0:
        print(stderr)
        return None
    return json.loads(stdout.split('\n')[-2])


def profile_ncu(config):
    config['gpu_id'] = GPU_ID
    config['algo'] = ALGO
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    details_folder = os.path.splitext(LOG_PATH)[0]
    if not os.path.exists(details_folder):
        os.makedirs(details_folder)
    exp_id = '_'.join([str(config[x]) for x in FEATURES])
    ncu_args = f'--set full --page=details --csv --print-summary per-kernel'
    ncu_cmd = f'ncu {ncu_args} {python_cmd} > {details_folder}/{exp_id}.csv'
    stdout, stderr = run_cmd(ncu_cmd)
    if len(stderr) > 0:
        print(stderr)
        return None
    return {}


if __name__ == '__main__':
    first_record = True
    for i, row in pd.read_csv(CONFIG_PATH).iterrows():
        config = row.to_dict()
        print(f'#{i}: {config}')
        results = {
            'energy': profile_energy,
            'latency': profile_latency,
            'ncu': profile_ncu,
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
