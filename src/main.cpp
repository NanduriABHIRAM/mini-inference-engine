#include "inference.hpp"

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
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

void validate_probabilities(const std::vector<float>& output, int batch, int classes) {
    if (output.size() != static_cast<std::size_t>(batch) * classes)
        throw std::runtime_error("Output shape mismatch.");
    for (int row = 0; row < batch; ++row) {
        double sum = 0.0;
        for (int col = 0; col < classes; ++col) {
            const float value = output[static_cast<std::size_t>(row) * classes + col];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
                throw std::runtime_error("Output contains an invalid probability.");
            sum += value;
        }
        if (std::abs(sum - 1.0) > 1e-5)
            throw std::runtime_error("Softmax row does not sum to one.");
    }
}
}

int main(int argc, char** argv) {
    try {
        int warmups = 3, trials = 7;
        fs::path csv = "results/benchmark.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (arg == "--warmups" && i + 1 < argc) warmups = std::stoi(argv[++i]);
            else if (arg == "--trials" && i + 1 < argc) trials = std::stoi(argv[++i]);
            else if (arg == "--help") {
                std::cout << "Usage: inference_[cpu|openmp] [--csv PATH] [--warmups N] [--trials N]\n";
                return 0;
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (warmups < 0 || trials < 1) throw std::runtime_error("warmups must be >=0; trials must be >=1.");

        const mie::MLP model = mie::make_reference_model();
        const int features = model.input_dim();
        const int classes = model.output_dim();
        {
            const auto check_input = mie::make_reference_input(4, features);
            std::vector<float> check_output(4 * classes), repeated(4 * classes);
            if (!mie::forward_cpu(model, check_input.data(), check_output.data(), 4) ||
                !mie::forward_cpu(model, check_input.data(), repeated.data(), 4))
                throw std::runtime_error("Sequential correctness pass failed.");
            validate_probabilities(check_output, 4, classes);
            if (mie::compare_outputs(check_output.data(), repeated.data(), check_output.size()).max_abs > 1e-7)
                throw std::runtime_error("Sequential inference is not repeatable.");
            std::cout << "Sequential correctness: PASS (shape, finite probabilities, softmax sums, repeatability)\n";
        }
#ifdef ENABLE_OPENMP
        {
            const auto check_input = mie::make_reference_input(4, features);
            std::vector<float> reference(4 * classes), parallel(4 * classes);
            if (!mie::forward_cpu(model, check_input.data(), reference.data(), 4) ||
                !mie::forward_cpu_omp(model, check_input.data(), parallel.data(), 4) ||
                mie::compare_outputs(reference.data(), parallel.data(), reference.size()).max_abs > 1e-5)
                throw std::runtime_error("Sequential and OpenMP outputs differ beyond 1e-5.");
            validate_probabilities(parallel, 4, classes);
            std::cout << "Sequential vs OpenMP correctness: PASS\n";
        }
#endif
        auto run = [&](const std::vector<float>& input, std::vector<float>& output, int batch) {
#ifdef ENABLE_OPENMP
            return mie::forward_cpu_omp(model, input.data(), output.data(), batch);
#else
            return mie::forward_cpu(model, input.data(), output.data(), batch);
#endif
        };
#ifdef ENABLE_OPENMP
        const std::string backend = "openmp_cpu";
        std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << "\n";
#else
        const std::string backend = "sequential_cpu";
#endif
        if (!csv.parent_path().empty()) fs::create_directories(csv.parent_path());
        const bool header = !fs::exists(csv) || fs::file_size(csv) == 0;
        std::ofstream out(csv, std::ios::app);
        if (!out) throw std::runtime_error("Cannot open CSV output: " + csv.string());
        if (header) out << "backend,batch_size,features,output_classes,warmup_runs,timed_runs,median_latency_ms,throughput_samples_per_sec,status\n";

        std::cout << "Shared workload: 512 -> 256 -> 256 -> 10, FP32\n";
        for (const int batch : {1, 16, 64, 256, 1024}) {
            const auto input = mie::make_reference_input(batch, features);
            std::vector<float> output(static_cast<std::size_t>(batch) * classes);
            for (int i = 0; i < warmups; ++i)
                if (!run(input, output, batch)) throw std::runtime_error("Inference failed.");
            std::vector<double> times;
            for (int i = 0; i < trials; ++i) {
                const auto start = Clock::now();
                if (!run(input, output, batch)) throw std::runtime_error("Inference failed.");
                times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            }
            const double ms = median(std::move(times));
            const double throughput = ms > 0.0 ? 1000.0 * batch / ms : 0.0;
            std::cout << backend << " batch=" << batch << " latency_ms=" << std::fixed
                      << std::setprecision(4) << ms << " throughput_samples_s=" << throughput << '\n';
            out << backend << ',' << batch << ',' << features << ',' << classes << ',' << warmups
                << ',' << trials << ',' << std::setprecision(10) << ms << ',' << throughput << ",measured\n";
        }
        std::cout << "CSV saved to: " << csv.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
