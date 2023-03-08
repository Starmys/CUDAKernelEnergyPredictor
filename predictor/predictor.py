import os
import glob
from typing import Dict

import numpy as np
import xgboost as xgb


MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
XGB_PARAMS = {
    'eta': 0.2,
    'gamma': 0.001,
    'n_estimators': 200,
    'max_depth': 10,
    'min_child_weight': 0,
    'seed': 2022,
}
ALGORITHMS = {
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM': 0,
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM': 1,
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM': 2,
    # 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT': 3,
    # 'CUDNN_CONVOLUTION_FWD_ALGO_FFT': 4,
    # 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING': 5,
    # 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD': 6,
    # 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED': 7,
}
COMPUTE_CAPABILITIES = {
    'P100': 60,
    '1080': 61,
    'V100': 70,
    '2080ti': 75,
}


class ONNXConvKernelEnergyPredictor(object):

    def __init__(self, device_type: str = 'V100'):
        self._cc = COMPUTE_CAPABILITIES[device_type]
        model_paths = glob.glob(os.path.join(MODEL_DIR, '*.model'))
        self._predictors: Dict[str, xgb.XGBRegressor] = {}
        for model_path in model_paths:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.load_model(model_path)
            target = os.path.splitext(os.path.split(model_path)[-1])[0]
            self._predictors[target] = model

    def _predict_energy(
        self,
        image_size: int,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        algo_enum: int,
        target: str,
    ) -> float:
        params = out_channels * (np.square(kernel_size) * in_channels / groups + 1)
        flops = np.square(image_size // stride)
        inputs = [[
            in_channels, out_channels, kernel_size, stride, groups, image_size,
            params, flops,
            algo_enum, self._cc,
        ]]
        pred = self._predictors[target].predict(inputs).item()
        if target != 'power':
            pred = np.exp2(pred)
        return pred

    def predict(
        self,
        image_size: int,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
    ) -> Dict[str, Dict[str, float]]:
        return {
            algo_name: {
                target: self._predict_energy(
                    image_size, kernel_size, in_channels, out_channels, stride, groups,
                    algo_enum, target,
                )
                for target in self._predictors.keys()
            }
            for algo_name, algo_enum in ALGORITHMS.items()
        }


if __name__ == '__main__':
    predictor = ONNXConvKernelEnergyPredictor('2080ti')
    for algo_name, algo_results in predictor.predict(14, 3, 128, 256, 1, 16).items():
        print(f'{algo_name}: {algo_results}')
