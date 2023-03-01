#include "run_onnx_model.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <cassert>
#include <algorithm>

#include <onnxruntime_c_api.h>
#include <cuda_energy_profiler.h>

#define PrintVar(x) (std::cout << #x << " = " << x << std::endl)

std::vector<double> run_onnx_model(
    const std::string& strModel,
    const ONNXRunConfig& cfg,
    const std::string& algoPresetPath
) {
    // TODO: param check for strModel

    // get api
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    assert(api != NULL);

    onnxruntime::profiling::GPUInspector& gpu_ins = onnxruntime::profiling::GPUInspector::Instance();
    bool success = gpu_ins.Init(cfg.gpu_id, cfg.gpu_sampling_interval);
    if(!success)
    {
        std::cerr << "Initialize GPUInspector failed." << std::endl;
        return std::vector<double>();
    }

    // create environment
    OrtEnv *env;
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "run_ort", &env);
    assert(env != NULL);

    // create session options
    OrtSessionOptions *session_options;
    api->CreateSessionOptions(&session_options);
    assert(session_options != NULL);
    api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);

    if(!cfg.optimized_model_save_path.empty())
    {
        std::cout << "Optimized model output set to: " << cfg.optimized_model_save_path << std::endl;
        api->SetOptimizedModelFilePath(session_options, cfg.optimized_model_save_path.c_str());
    }
    if(!cfg.profile_save_path.empty())
    {
        std::cout << "Profiling output set to: " << cfg.profile_save_path << std::endl;
        api->EnableProfiling(session_options, cfg.profile_save_path.c_str());
    }

    // enable cuda
    auto enable_cuda = [&](OrtSessionOptions* session_options)->int {
        // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
        OrtCUDAProviderOptions o;
        // Here we use memset to initialize every field of the above data struct to zero.
        memset(&o, 0, sizeof(o));
        // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
        // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
        o.device_id = 0;
        o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchPreset;
        o.algo_preset_file = const_cast<char*>(algoPresetPath.c_str());
        o.gpu_mem_limit = SIZE_MAX;
        OrtStatus* onnx_status = api->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
        if (onnx_status != NULL) {
            const char* msg = api->GetErrorMessage(onnx_status);
            std::cerr << msg << std::endl;
            api->ReleaseStatus(onnx_status);
            return -1;
        }
        return 0;
    };
    int ret = enable_cuda(session_options);
    if(ret)
    {
        std::cerr << "CUDA is not available." << std::endl;
    }
    else
    {
        std::cout << "CUDA is enabled." << std::endl;
    }

    // create session and load model
    OrtSession *session;
    api->CreateSession(env, strModel.c_str(), session_options, &session);
    std::cout << "Load model finished." << std::endl;

    // create allocator
    OrtAllocator *allocator;
    api->GetAllocatorWithDefaultOptions(&allocator);
    assert(allocator != NULL);

    // setup input and output
    size_t n_inputs = 0, n_outputs = 0;
    api->SessionGetInputCount(session, &n_inputs);
    api->SessionGetOutputCount(session, &n_outputs);
    OrtValue** ort_input_values = new OrtValue*[n_inputs];
    OrtValue** ort_output_values = new OrtValue*[n_outputs];
    char** input_names = new char*[n_inputs];
    char** output_names = new char*[n_outputs];
    for(size_t i = 0; i < n_inputs; i++)
    {
        api->SessionGetInputName(session, i, allocator, &input_names[i]);

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetInputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);

        enum ONNXTensorElementDataType element_type;
        api->GetTensorElementType(tensor_info, &element_type);
        size_t n_dims;
        api->GetDimensionsCount(tensor_info, &n_dims);
        int64_t *dim_values = new int64_t[n_dims];
        api->GetDimensions(tensor_info, dim_values, n_dims);

        api->CreateTensorAsOrtValue(allocator, dim_values, n_dims, element_type, &ort_input_values[i]);
        api->ReleaseTypeInfo(type_info);
    }
    for(size_t i = 0; i < n_outputs; i++)
    {
        api->SessionGetOutputName(session, i, allocator, &output_names[i]);

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetOutputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);

        enum ONNXTensorElementDataType element_type;
        api->GetTensorElementType(tensor_info, &element_type);
        size_t n_dims;
        api->GetDimensionsCount(tensor_info, &n_dims);
        int64_t *dim_values = new int64_t[n_dims];
        api->GetDimensions(tensor_info, dim_values, n_dims);

        api->CreateTensorAsOrtValue(allocator, dim_values, n_dims, element_type, &ort_output_values[i]);
        api->ReleaseTypeInfo(type_info);
    }

    // warming up
    if(cfg.warm_up_repeat)
    {
        std::cout << "Warming up begin." << std::endl;
        for(int i = 0; i < cfg.warm_up_repeat; i++)
        {
            api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
        }
        std::cout << "Warming up finished." << std::endl;
    }

    // run inference
    std::cout << "Run begin." << std::endl;
    unsigned int repeat = cfg.num_repeat;
    onnxruntime::profiling::Timer run_timer;
    gpu_ins.StartInspect();
    run_timer.start();
    for(int i = 0; i < repeat; i++)
    {
        api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
        double time_elapsed = run_timer.getElapsedTimeInSec();
        if(time_elapsed >= cfg.time_limit)
        {
            std::cout << "early stop due to exceeding time limit, actual repeat times: " << i + 1 << std::endl;
            repeat = i + 1;
        }
    }
    run_timer.stop();
    gpu_ins.StopInspect();
    std::cout << "Run finished." << std::endl;

    // destroy the structures
    api->ReleaseSessionOptions(session_options);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    // api->ReleaseAllocator(allocator);

    // display log info
    std::cout << "NumDevices = " << gpu_ins.NumDevices() << std::endl;
    std::cout << "NumInspectedDevices = " << gpu_ins.NumInspectedDevices() << std::endl;
    PrintVar(repeat);

    // display energy and latency
    std::vector<double> energies;
    gpu_ins.CalculateEnergy(energies);
    std::cout << "energies(in vector) : ";
    for(double item : energies)
    {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "gpu latency = " << gpu_ins.GetDurationInSec() << " second" << std::endl;

    // export latency and energy
    if(!cfg.running_logs_save_path.empty())
    {
        std::ofstream f_log(cfg.running_logs_save_path);
        f_log << "model,repeat,latency,energy" << std::endl;
        f_log << std::fixed;
        f_log << strModel << "," << repeat << "," << gpu_ins.GetDurationInSec();
        for(double item : energies)
        {
            f_log << "," << item;
        }
        f_log << std::endl;
        f_log.close();
    }

    // export GPU readings
    if(!cfg.gpu_readings_csv_save_path.empty())
    {
        std::ofstream f_gpu(cfg.gpu_readings_csv_save_path);
        f_gpu << "gpu_id,timestamp,used_memory,power,temperature,memory_util,gpu_util" << std::endl;
        f_gpu << std::fixed;
        std::vector<unsigned int> device_ids;
        gpu_ins.InspectedDeviceIds(device_ids);
        std::sort(device_ids.begin(), device_ids.end());
        for(unsigned int gpu_id : device_ids)
        {
            std::vector<onnxruntime::profiling::GPUInspector::GPUInfo_t> gpu_readings;
            gpu_ins.ExportReadings(gpu_id, gpu_readings);
            for(const auto& info : gpu_readings)
            {
                f_gpu << gpu_id << "," << info.time_stamp << "," << info.used_memory_percent << "," << info.power_watt << "," 
                    << info.temperature << "," << info.memory_util << "," << info.gpu_util << std::endl;
            }
        }
        f_gpu.close();
    }

    return {gpu_ins.GetDurationInSec() / repeat, gpu_ins.CalculateEnergy(cfg.gpu_id) / repeat};
}
