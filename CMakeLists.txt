cmake_minimum_required(VERSION 3.16)

project(mini_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Sequential CPU backend
add_executable(inference_cpu src/main.cpp)

# Parallel OpenMP CPU backend
add_executable(inference_openmp src/main.cpp)

target_compile_definitions(
    inference_openmp PRIVATE ENABLE_OPENMP
)

target_link_libraries(
    inference_openmp PRIVATE OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(inference_cpu PRIVATE /W4)
    target_compile_options(inference_openmp PRIVATE /W4)
else()
    target_compile_options(inference_cpu PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(inference_openmp PRIVATE -Wall -Wextra -Wpedantic)
endif()
