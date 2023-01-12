import os
import glob
from typing import Dict

import numpy as np
import xgboost as xgb


MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
XGB_PARAMS = {
    'eta': 0.2,
    'gamma': 0.001,
    'n_estimators': 100,
    'max_depth': 10,
    'min_child_weight': 0,
    'seed': 2022,
}
ALGORITHMS = [
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD',
]


class ONNXConvKernelEnergyPredictor(object):

    def __init__(self, device_type: str = 'V100'):
        model_paths = glob.glob(os.path.join(MODEL_DIR, f'{device_type}_*.model'))
        assert len(model_paths) > 0, f'No predictor found for {device_type}'
        self._predictors: Dict[str, xgb.XGBRegressor] = {}
        for model_path in model_paths:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.load_model(model_path)
            target = os.path.splitext(os.path.split(model_path)[-1])[0].split('_')[1]
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
        inputs = [[in_channels, out_channels, kernel_size, stride, groups, image_size, algo_enum]]
        return np.exp2(self._predictors[target].predict(inputs)).item()

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
            for algo_enum, algo_name in enumerate(ALGORITHMS)
        }


if __name__ == '__main__':
    predictor = ONNXConvKernelEnergyPredictor()
    for algo_name, algo_results in predictor.predict(14, 3, 128, 256, 1, 16).items():
        print(f'{algo_name}: {algo_results}')
