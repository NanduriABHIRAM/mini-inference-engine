cmake_minimum_required(VERSION 3.16)

project(mini_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

option(MIE_ENABLE_OPENMP "Build the OpenMP backend when the compiler supports it" ON)
if(MIE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

# Sequential CPU backend
add_executable(inference_cpu
    src/main.cpp
    src/cpu_infer.cpp
    src/model.cpp
)

# Parallel OpenMP CPU backend
add_executable(inference_openmp
    src/main.cpp
    src/cpu_infer.cpp
    src/model.cpp
)

# The Streamlit application invokes this target for individual inference runs.
# It always supports sequential CPU and OpenMP; CUDA is added below when found.
add_executable(inference_runner
    src/inference_runner.cpp
    src/cpu_infer.cpp
    src/model.cpp
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(inference_openmp PRIVATE ENABLE_OPENMP)
    target_link_libraries(inference_openmp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(inference_runner PRIVATE OpenMP::OpenMP_CXX)
else()
    # Apple Clang does not ship OpenMP by default.  Keep the target available
    # as an explicitly labelled sequential fallback rather than making a CPU-
    # only checkout fail at CMake configure time.
    target_compile_definitions(inference_openmp PRIVATE MIE_OPENMP_UNAVAILABLE=1)
    message(STATUS "OpenMP C++ support not found; inference_openmp will be a sequential fallback.")
endif()

if(MSVC)
    target_compile_options(inference_cpu PRIVATE /W4)
    target_compile_options(inference_openmp PRIVATE /W4)
    target_compile_options(inference_runner PRIVATE /W4)
else()
    target_compile_options(inference_cpu PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(inference_openmp PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(inference_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CUDA is optional so CPU-only machines can still configure and build.
if(CMAKE_CUDA_COMPILER AND NOT CMAKE_VERSION VERSION_LESS 3.18)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(MIE_CUDA_ARCHITECTURES "86" CACHE STRING
        "CUDA compute capabilities to compile for (for example: 75;86;89 or native)")

    # WSL exposes the Windows NVIDIA driver through this library.  A Linux
    # NVIDIA driver installed in the distribution can otherwise be selected
    # first by the dynamic loader, even though it cannot use WSL's /dev/dxg
    # device.  Keep this configurable for WSL installations with a custom
    # mount point.
    set(MIE_WSL_CUDA_DRIVER_DIR "/usr/lib/wsl/lib" CACHE PATH
        "Directory containing WSL's libcuda.so.1 bridge")
    set(mie_cuda_driver CUDA::cuda_driver)
    set(mie_cuda_driver_rpath "")
    if(UNIX AND EXISTS "${MIE_WSL_CUDA_DRIVER_DIR}/libcuda.so.1")
        add_library(mie_wsl_cuda_driver SHARED IMPORTED GLOBAL)
        set_target_properties(mie_wsl_cuda_driver PROPERTIES
            IMPORTED_LOCATION "${MIE_WSL_CUDA_DRIVER_DIR}/libcuda.so.1"
            IMPORTED_SONAME "libcuda.so.1"
        )
        set(mie_cuda_driver mie_wsl_cuda_driver)
        set(mie_cuda_driver_rpath "${MIE_WSL_CUDA_DRIVER_DIR}")
        message(STATUS "Using the WSL CUDA driver bridge: ${MIE_WSL_CUDA_DRIVER_DIR}/libcuda.so.1")
    endif()

    add_library(mie_cuda STATIC cuda/gpu_infer.cu)
    target_include_directories(mie_cuda PUBLIC src cuda)
    target_link_libraries(mie_cuda PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(mie_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MIE_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared
        POSITION_INDEPENDENT_CODE ON
    )

    add_executable(inference_cuda
        src/gpu_benchmark.cpp
        src/cpu_infer.cpp
        src/model.cpp
    )
    # gpu_benchmark.cpp calls the Driver API before the Runtime API.  This
    # makes libcuda a direct dependency, so the WSL bridge above is loaded
    # before libcudart attempts device discovery.
    target_link_libraries(inference_cuda PRIVATE mie_cuda ${mie_cuda_driver})
    set_target_properties(inference_cuda PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(inference_runner PRIVATE mie_cuda ${mie_cuda_driver})
    target_compile_definitions(inference_runner PRIVATE MIE_HAS_CUDA=1)
    set_target_properties(inference_runner PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    if(mie_cuda_driver_rpath)
        set_property(TARGET inference_cuda APPEND PROPERTY BUILD_RPATH
            "${mie_cuda_driver_rpath}")
        set_property(TARGET inference_runner APPEND PROPERTY BUILD_RPATH
            "${mie_cuda_driver_rpath}")
    endif()
elseif(CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA detected, but the GPU backend requires CMake 3.18 or newer.")
else()
    message(STATUS "CUDA compiler not found; skipping the optional GPU backend.")
endif()
