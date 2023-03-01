#include <string>
#include <vector>

struct ONNXRunConfig
{
    unsigned int num_repeat = 1;
    unsigned int warm_up_repeat = 0;
    unsigned int gpu_id = 0;
    float gpu_sampling_interval = 0.01;
    double time_limit = std::numeric_limits<double>::max();
    std::string optimized_model_save_path;
    std::string gpu_readings_csv_save_path;
    std::string running_logs_save_path;
    std::string profile_save_path;
};

std::vector<double> run_onnx_model(
    const std::string& strModel,
    const ONNXRunConfig& cfg,
    const std::string& algoPresetPath
);
