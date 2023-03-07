import os
import sys
import math
import datetime
from typing import Dict, Any

import pandas as pd

from onnx_conv import test_op


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
FEATURES = ['hw', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'groups']


class BatchProfiler(object):

    def __init__(self, mode: str = 'energy', gpu_id: int = 0, algo: int = 0):
        self._mode = mode
        self._algo = algo
        self._gpu_id = gpu_id
        config_set = {'energy': 'unique', 'latency': 'expand'}[mode]
        self._config_path = os.path.join(
            os.path.dirname(__file__),
            'search_space',
            f'regnet_convs_{config_set}.csv',
        )
        log_folder = os.path.join(os.path.dirname(__file__), 'logs', mode)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self._log_path = os.path.join(
            log_folder,
            f'{algo}-{datetime.date.today().strftime("%Y%m%d")}.csv',
        )
        self._profile = {
            'energy': self._profile_energy,
            'latency': self._profile_latency,
        }[self._mode]

    def _profile_config(self, num_warmups: int, num_iters: int, config: Dict[str, Any]):
        return test_op({
            'gpu_id': self._gpu_id,
            'algo': self._algo,
            'num_warmups': num_warmups,
            'num_iters': num_iters,
            **config
        })

    def _profile_energy(self, config):
        trial_results = self._profile_config(10, 100, config)
        if trial_results is None:
            return None
        num_iters = math.ceil(10 / trial_results['latency'])
        return self._profile_config(200, num_iters, config)

    def _profile_latency(self, config):
        return self._profile_config(200, 1000, config)

    def run(self):
        first_record = True
        for i, row in pd.read_csv(self._config_path).iterrows():
            config = row.to_dict()
            print(f'#{i}: {config}')
            results = self._profile(config)
            if results is None:
                continue
            data = config | results
            if first_record:
                with open(self._log_path, 'w') as f:
                    f.write(','.join(data.keys()) + '\n')
                first_record = False
            with open(self._log_path, 'a') as f:
                f.write(','.join([str(x) for x in data.values()]) + '\n')


if __name__ == '__main__':
    gpu_id = 0
    algo = 0
    for arg in sys.argv[1:]:
        k, v = arg[2:].split('=')
        if k.lower().startswith('gpu'):
            gpu_id = int(v)
        elif k.lower().startswith('algo'):
            algo = int(v)
    BatchProfiler('energy', gpu_id, algo).run()
