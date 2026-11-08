#include "gpu_infer.cuh"
#include "inference.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

template <typename Function>
double time_median(Function&& function, int warmups, int trials) {
    for (int i = 0; i < warmups; ++i) function();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));
    for (int i = 0; i < trials; ++i) {
        const auto start = Clock::now();
        function();
        const auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return median(std::move(times));
}

std::string driver_error(CUresult status) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);

    std::string result = name != nullptr ? name : "unknown CUDA driver error";
    if (message != nullptr) result += std::string(" (") + message + ')';
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int warmups = 3;
        int trials = 7;
        fs::path csv = "results/cuda_comparison.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (arg == "--warmups" && i + 1 < argc) warmups = std::stoi(argv[++i]);
            else if (arg == "--trials" && i + 1 < argc) trials = std::stoi(argv[++i]);
            else if (arg == "--help") {
                std::cout << "Usage: inference_cuda [--csv PATH] [--warmups N] [--trials N]\n";
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + arg);
            }
        }
        if (warmups < 0 || trials < 1)
            throw std::runtime_error("warmups must be >= 0; trials must be >= 1.");

        const CUresult driver_status = cuInit(0);
        if (driver_status != CUDA_SUCCESS) {
            throw std::runtime_error(
                "CUDA driver cannot access a GPU: " + driver_error(driver_status));
        }

        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            const char* reason = device_status == cudaSuccess
                                     ? "no CUDA device was found"
                                     : cudaGetErrorString(device_status);
            throw std::runtime_error(std::string("CUDA runtime cannot access a GPU: ") + reason);
        }

        const mie::MLP model = mie::make_reference_model();
        const int features = model.input_dim();
        fs::create_directories(csv.parent_path().empty() ? fs::path(".") : csv.parent_path());
        std::ofstream out(csv);
        if (!out) throw std::runtime_error("Cannot open CSV output: " + csv.string());
        out << "batch_size,cpu_median_ms,gpu_median_ms,gpu_speedup,cpu_samples_per_sec,"
               "gpu_samples_per_sec,max_abs_difference,rms_difference\n";

        std::cout << "CUDA inference comparison (CPU reference vs CUDA/cuBLAS)\n"
                  << "Synthetic network: 512 -> 256 -> 256 -> 10, FP32\n"
                  << "GPU timings include host/device transfers and per-call setup.\n\n"
                  << std::left << std::setw(12) << "Batch"
                  << std::right << std::setw(14) << "CPU ms"
                  << std::setw(14) << "GPU ms"
                  << std::setw(14) << "Speedup"
                  << std::setw(16) << "Max abs diff" << '\n';

        for (const int batch : {1, 16, 64, 256, 1024}) {
            const std::vector<float> input = mie::make_reference_input(batch, features);
            std::vector<float> cpu_output(static_cast<std::size_t>(batch) * model.output_dim());
            std::vector<float> gpu_output(cpu_output.size());

            if (!mie::forward_cpu(model, input.data(), cpu_output.data(), batch))
                throw std::runtime_error("CPU reference inference failed.");
            if (!mie::forward_gpu(model, input.data(), gpu_output.data(), batch))
                throw std::runtime_error("CUDA inference failed; check CUDA device and runtime.");

            const mie::DiffStats diff = mie::compare_outputs(
                cpu_output.data(), gpu_output.data(), cpu_output.size());
            constexpr double tolerance = 1e-4;
            if (diff.max_abs > tolerance)
                throw std::runtime_error("CPU and CUDA outputs differ beyond tolerance (max abs=" +
                                         std::to_string(diff.max_abs) + ").");

            const double cpu_ms = time_median([&] {
                if (!mie::forward_cpu(model, input.data(), cpu_output.data(), batch))
                    throw std::runtime_error("CPU reference inference failed.");
            }, warmups, trials);
            const double gpu_ms = time_median([&] {
                if (!mie::forward_gpu(model, input.data(), gpu_output.data(), batch))
                    throw std::runtime_error("CUDA inference failed during timing.");
            }, warmups, trials);
            const double speedup = cpu_ms / gpu_ms;
            const double cpu_throughput = 1000.0 * batch / cpu_ms;
            const double gpu_throughput = 1000.0 * batch / gpu_ms;

            std::cout << std::left << std::setw(12) << batch
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(14) << cpu_ms << std::setw(14) << gpu_ms
                      << std::setprecision(2) << std::setw(14) << speedup
                      << std::setprecision(7) << std::setw(16) << diff.max_abs << '\n';
            out << batch << ',' << std::setprecision(10) << cpu_ms << ',' << gpu_ms << ','
                << speedup << ',' << cpu_throughput << ',' << gpu_throughput << ','
                << diff.max_abs << ',' << diff.rms << '\n';
        }
        std::cout << "\nOutput comparison: PASS (max absolute error <= 1e-4)\n"
                  << "CSV saved to: " << csv.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
