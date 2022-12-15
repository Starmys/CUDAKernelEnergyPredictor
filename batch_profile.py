import os
import math
import json
import datetime
import subprocess

import pandas as pd


GPU_ID = 7
ALGO = 'gemm'  # gemm/7, igemm/6, ipgemm/5, fft/4, fftt/3, wino/2
MODE = 'latency'

if MODE == 'energy':
    CONFIG_PATH = os.path.join('search_space', 'regnet_convs_unique.csv')
    LOG_FOLDER = os.path.join('logs', MODE)
elif MODE == 'latency':
    CONFIG_PATH = os.path.join('search_space', 'regnet_convs_expand.csv')
    LOG_FOLDER = os.path.join('logs', MODE)
else:
    raise ValueError(f'mode not supported: {MODE}')

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
LOG_PATH = os.path.join(LOG_FOLDER, f'{ALGO}-{datetime.date.today().strftime("%Y%m%d")}.csv')


def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return stdout, stderr


def profile_energy(config):
    # prefix_cmd = f'conda activate ort-conv-{ALGO}'
    config['gpu_id'] = GPU_ID
    config['sample_num'] = 10
    config['gpu_sampler'] = 0
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    stdout, stderr = run_cmd(python_cmd)
    if len(stderr) > 0:
        print(stderr)
        return None
    config['sample_num'] = math.ceil(10 / json.loads(stdout)['latency'])
    config['gpu_sampler'] = 1
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    stdout, stderr = run_cmd(python_cmd)
    return json.loads(stdout)


def profile_latency(config):
    config['gpu_id'] = GPU_ID
    config['sample_num'] = 100
    config['gpu_sampler'] = 0
    python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
    python_cmd = f'python onnx_conv.py {python_args}'
    stdout, stderr = run_cmd(python_cmd)
    if len(stderr) > 0:
        print(stderr)
        return None
    return json.loads(stdout)


# ncu_args = f'--set full --page=details --csv --print-summary per-kernel'
# ncu_cmd = f'ncu {ncu_args} {python_cmd} > {LOG_FOLDER}/{str(i).zfill(4)}.csv'


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
