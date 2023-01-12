```python
from predictor import ONNXConvKernelEnergyPredictor
predictor = ONNXConvKernelEnergyPredictor('V100')
results = predictor.predict(image_size, kernel_size, in_channels, out_channels, stride, groups)
```