cmake_minimum_required(VERSION 3.4...3.18)
project(onnx_model_runner_backend)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -fPIC")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(USE_CUDA)
add_subdirectory(cuda_energy_profiler)

# set(ONNXRUNTIME_ROOTDIR "/home/t-chengluo/donglinb/Projects/onnxruntime/build/Linux/install")
set(ONNXRUNTIME_ROOTDIR "/data2/cz/onnxruntime-conv-preset/build/Linux/install")
include_directories(
    "${ONNXRUNTIME_ROOTDIR}/include"
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
list(APPEND EXTERNAL_LIBS 
    onnxruntime 
    onnxruntime_providers_shared 
    libonnxruntime_providers_cuda.so 
)

add_library(run_onnx_model src/run_onnx_model.cpp)
target_link_libraries(run_onnx_model ${EXTERNAL_LIBS} cuda_energy_profiler)

add_subdirectory(pybind11)
pybind11_add_module(onnx_model_runner_backend src/main.cpp)
target_link_libraries(onnx_model_runner_backend PRIVATE run_onnx_model)

# https://stackoverflow.com/questions/6594796/how-do-i-make-cmake-output-into-a-bin-dir
set_target_properties(onnx_model_runner_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(cmake_example
#                            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

