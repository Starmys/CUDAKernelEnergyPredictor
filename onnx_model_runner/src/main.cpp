#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "run_onnx_model.h"

namespace py = pybind11;

PYBIND11_MODULE(onnx_model_runner_backend, m)
{
    py::class_<ONNXRunConfig>(m, "ONNXRunConfig")
        .def(py::init<>())
        .def_readwrite("num_repeat", &ONNXRunConfig::num_repeat)
        .def_readwrite("warm_up_repeat", &ONNXRunConfig::warm_up_repeat)
        .def_readwrite("gpu_id", &ONNXRunConfig::gpu_id)
        .def_readwrite("gpu_sampling_interval", &ONNXRunConfig::gpu_sampling_interval)
        .def_readwrite("time_limit", &ONNXRunConfig::time_limit)
        .def_readwrite("optimized_model_save_path", &ONNXRunConfig::optimized_model_save_path)
        .def_readwrite("gpu_readings_csv_save_path", &ONNXRunConfig::gpu_readings_csv_save_path)
        .def_readwrite("running_logs_save_path", &ONNXRunConfig::running_logs_save_path)
        .def_readwrite("profile_save_path", &ONNXRunConfig::profile_save_path);
    m.def("run_onnx_model", &run_onnx_model, "Run ONNX Models.");
}
