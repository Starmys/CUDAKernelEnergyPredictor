import json
import subprocess

import numpy as np

config = {
    'in_channels': 1360,
    'out_channels': 1360,
    'kernel_size': 1,
    'stride': 1,
    'groups': 1,
    'hw': 7,
    'gpu_id': 0,
    'num_warmups': 200,
    'num_iters': 1000,
    'gpu_sampler': 0,
}
python_args = ' '.join([f'--{k}={v}' for k, v in config.items()])
python_cmd = f'python onnx_conv.py {python_args}'

latency_list = []
for _ in range(10):
    p = subprocess.Popen(python_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    latency_list.append(json.loads(stdout.decode("utf-8"))['latency'])

m = np.mean(latency_list)
e = np.std(latency_list)
print(config['num_warmups'], config['num_iters'], m, e, e / m)
