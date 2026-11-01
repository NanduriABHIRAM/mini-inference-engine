#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Layer {
    int in_features, out_features;
    std::vector<float> weights; // [out_features][in_features], row-major
    std::vector<float> bias;
};
struct Model { std::vector<Layer> layers; };

std::vector<float> random_values(std::size_t n, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

Model make_model() {
    // Deterministic synthetic weights for a repeatable compute workload, not a trained classifier.
    std::mt19937 rng(12345);
    const int dims[] = {512, 256, 256, 10};
    Model m;
    for (int i = 0; i < 3; ++i) {
        Layer l{dims[i], dims[i + 1], {}, {}};
        float scale = std::sqrt(2.0f / l.in_features);
        l.weights = random_values(static_cast<std::size_t>(l.in_features) * l.out_features,
                                  rng, -scale, scale);
        l.bias = random_values(static_cast<std::size_t>(l.out_features), rng, -0.05f, 0.05f);
        m.layers.push_back(std::move(l));
    }
    return m;
}

std::vector<float> make_input(int batch, int features) {
    std::mt19937 rng(67890 + batch);
    return random_values(static_cast<std::size_t>(batch) * features, rng, -1.0f, 1.0f);
}

void softmax(float* row, int n) {
    float max_v = row[0];
    for (int i = 1; i < n; ++i) max_v = std::max(max_v, row[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        row[i] = std::exp(row[i] - max_v);
        sum += row[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum))
        throw std::runtime_error("Softmax normalization failed.");
    for (int i = 0; i < n; ++i) row[i] = static_cast<float>(row[i] / sum);
}

std::vector<float> forward_cpu(const Model& model, const std::vector<float>& input, int batch) {
    if (model.layers.empty()) throw std::runtime_error("Model has no layers.");
    int current_features = model.layers.front().in_features;
    if (input.size() != static_cast<std::size_t>(batch) * current_features)
        throw std::runtime_error("Input dimensions do not match model.");

    std::vector<float> current = input;
    for (std::size_t li = 0; li < model.layers.size(); ++li) {
        const Layer& l = model.layers[li];
        if (l.in_features != current_features) throw std::runtime_error("Layer dimension mismatch.");
        std::vector<float> next(static_cast<std::size_t>(batch) * l.out_features, 0.0f);

        for (int b = 0; b < batch; ++b) {
            const float* x = current.data() + static_cast<std::size_t>(b) * l.in_features;
            float* y = next.data() + static_cast<std::size_t>(b) * l.out_features;
            for (int o = 0; o < l.out_features; ++o) {
                double acc = l.bias[o];
                const float* w = l.weights.data() + static_cast<std::size_t>(o) * l.in_features;
                for (int k = 0; k < l.in_features; ++k)
                    acc += static_cast<double>(x[k]) * w[k];
                y[o] = static_cast<float>(acc);
            }
            if (li + 1 == model.layers.size()) {
                softmax(y, l.out_features);
            } else {
                for (int o = 0; o < l.out_features; ++o) y[o] = std::max(0.0f, y[o]);
            }
        }
        current = std::move(next);
        current_features = l.out_features;
    }
    return current;
}

#ifdef ENABLE_OPENMP
std::vector<float> forward_openmp(
    const Model& model,
    const std::vector<float>& input,
    int batch
) {
    if (model.layers.empty())
        throw std::runtime_error("Model has no layers.");

    int current_features = model.layers.front().in_features;

    if (input.size() != static_cast<std::size_t>(batch) * current_features)
        throw std::runtime_error("Input dimensions do not match model.");

    std::vector<float> current = input;

    for (std::size_t li = 0; li < model.layers.size(); ++li) {
        const Layer& l = model.layers[li];

        if (l.in_features != current_features)
            throw std::runtime_error("Layer dimension mismatch.");

        std::vector<float> next(
            static_cast<std::size_t>(batch) * l.out_features,
            0.0f
        );

        // Each thread processes different samples in the batch.
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < batch; ++b) {
            const float* x =
                current.data() + static_cast<std::size_t>(b) * l.in_features;

            float* y =
                next.data() + static_cast<std::size_t>(b) * l.out_features;

            for (int o = 0; o < l.out_features; ++o) {
                double acc = l.bias[o];

                const float* w =
                    l.weights.data() + static_cast<std::size_t>(o) * l.in_features;

                for (int k = 0; k < l.in_features; ++k) {
                    acc += static_cast<double>(x[k]) * w[k];
                }

                y[o] = static_cast<float>(acc);
            }

            if (li + 1 == model.layers.size()) {
                softmax(y, l.out_features);
            } else {
                for (int o = 0; o < l.out_features; ++o) {
                    y[o] = std::max(0.0f, y[o]);
                }
            }
        }

        current = std::move(next);
        current_features = l.out_features;
    }

    return current;
}

#endif // ENABLE_OPENMP

void correctness_checks(const Model& model) {
    const int batch = 4;
    const int classes = model.layers.back().out_features;
    auto input = make_input(batch, model.layers.front().in_features);
    auto output = forward_cpu(model, input, batch);
    if (output.size() != static_cast<std::size_t>(batch) * classes)
        throw std::runtime_error("FAIL: output shape.");
    for (float x : output) {
        if (!std::isfinite(x)) throw std::runtime_error("FAIL: non-finite output.");
        if (x < 0.0f || x > 1.0f) throw std::runtime_error("FAIL: probability outside [0,1].");
    }
    for (int b = 0; b < batch; ++b) {
        double sum = 0.0;
        for (int j = 0; j < classes; ++j)
            sum += output[static_cast<std::size_t>(b) * classes + j];
        if (std::abs(sum - 1.0) > 1e-5) throw std::runtime_error("FAIL: softmax row sum.");
    }
    auto again = forward_cpu(model, input, batch);
    for (std::size_t i = 0; i < output.size(); ++i)
        if (std::abs(output[i] - again[i]) > 1e-7f)
            throw std::runtime_error("FAIL: repeatability.");
    std::cout << "Correctness checks: PASS (shape, finite values, range, softmax sums, repeatability)\n";
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto n = v.size();
    if (n == 0) throw std::runtime_error("No timing samples.");
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

void benchmark(const Model& model, const std::vector<int>& batches,
               const fs::path& csv, int warmups, int trials) {
    if (!csv.parent_path().empty()) fs::create_directories(csv.parent_path());
    const bool needs_header = !fs::exists(csv) || fs::file_size(csv) == 0;
    std::ofstream out(csv, std::ios::app);
    if (!out) throw std::runtime_error("Cannot open CSV output.");
    if (needs_header)
        out << "backend,batch_size,features,output_classes,warmup_runs,timed_runs,median_latency_ms,throughput_samples_per_sec,status\n";

    const int features = model.layers.front().in_features;
    const int classes = model.layers.back().out_features;
    volatile std::size_t output_guard = 0;

    std::cout << "\nSequential CPU benchmark (measurements from this run)\n";
    std::cout << std::left << std::setw(12) << "Batch" << std::right
              << std::setw(18) << "Median ms" << std::setw(25) << "Samples/second" << "\n";

    for (int batch : batches) {
        auto input = make_input(batch, features);
        for (int i = 0; i < warmups; ++i) {
            auto ignored = forward_cpu(model, input, batch);
            output_guard += ignored.size();
        }
        std::vector<double> times;
        for (int i = 0; i < trials; ++i) {
            auto start = Clock::now();
            auto output = forward_cpu(model, input, batch);
            auto end = Clock::now();
            output_guard += output.size();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        const double ms = median(times);
        const double throughput = ms > 0.0 ? 1000.0 * batch / ms : 0.0;
        std::cout << std::left << std::setw(12) << batch << std::right
                  << std::fixed << std::setprecision(3) << std::setw(18) << ms
                  << std::setprecision(1) << std::setw(25) << throughput << "\n";
        out << "sequential_cpu," << batch << "," << features << "," << classes << ","
            << warmups << "," << trials << "," << std::setprecision(10) << ms << ","
            << throughput << ",measured\n";
    }
    (void)output_guard;
    std::cout << "CSV saved to: " << csv.string() << "\n";
}

#ifdef ENABLE_OPENMP
void benchmark_openmp(
    const Model& model,
    const std::vector<int>& batches,
    const fs::path& csv,
    int warmups,
    int trials
) {
    if (!csv.parent_path().empty())
        fs::create_directories(csv.parent_path());

    const bool needs_header = !fs::exists(csv) || fs::file_size(csv) == 0;

    std::ofstream out(csv, std::ios::app);

    if (!out)
        throw std::runtime_error("Cannot open CSV output.");

    if (needs_header) {
        out << "backend,batch_size,features,output_classes,"
               "warmup_runs,timed_runs,median_latency_ms,"
               "throughput_samples_per_sec,status\n";
    }

    int features = model.layers.front().in_features;
    int classes = model.layers.back().out_features;

    volatile std::size_t output_guard = 0;

    std::cout << "\nOpenMP CPU benchmark\n";
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << "\n";

    std::cout << std::left << std::setw(12) << "Batch"
              << std::right << std::setw(18) << "Median ms"
              << std::setw(25) << "Samples/second" << "\n";

    for (int batch : batches) {
        auto input = make_input(batch, features);

        for (int i = 0; i < warmups; ++i) {
            auto ignored = forward_openmp(model, input, batch);
            output_guard += ignored.size();
        }

        std::vector<double> times;

        for (int i = 0; i < trials; ++i) {
            auto start = Clock::now();

            auto output = forward_openmp(model, input, batch);

            auto end = Clock::now();

            output_guard += output.size();

            times.push_back(
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count()
            );
        }

        const double ms = median(times);
        const double throughput = 1000.0 * batch / ms;

        std::cout << std::left << std::setw(12) << batch
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(18) << ms
                  << std::setprecision(1)
                  << std::setw(25) << throughput << "\n";

        out << "openmp_cpu," << batch << "," << features << ","
            << classes << "," << warmups << "," << trials << ","
            << std::setprecision(10) << ms << ","
            << throughput << ",measured\n";
    }

    (void)output_guard;

    std::cout << "CSV saved to: " << csv.string() << "\n";
}

#endif // ENABLE_OPENMP

int main(int argc, char** argv) {
    try {
        int warmups = 3, trials = 7;
        fs::path csv = "results/benchmark.csv";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (arg == "--warmups" && i + 1 < argc) warmups = std::stoi(argv[++i]);
            else if (arg == "--trials" && i + 1 < argc) trials = std::stoi(argv[++i]);
            else if (arg == "--help") {
                std::cout << "Usage: inference_[cpu|openmp] [--csv PATH] [--warmups N] [--trials N]\n";
                return 0;
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (warmups < 0 || trials < 1) throw std::runtime_error("warmups must be >=0; trials must be >=1.");
Model model = make_model();

std::cout << "Synthetic workload: 512 -> 256 -> 256 -> 10, FP32\n";
std::cout << "Weights/inputs are deterministic synthetic values, not a trained classifier.\n";

#ifdef ENABLE_OPENMP

    std::cout << "Mini Inference Engine — Phase 2: OpenMP CPU\n";

    // Check OpenMP output against sequential output.
    auto input = make_input(4, model.layers.front().in_features);

    auto sequential_output = forward_cpu(model, input, 4);
    auto parallel_output = forward_openmp(model, input, 4);

    if (sequential_output.size() != parallel_output.size())
        throw std::runtime_error("OpenMP output shape mismatch.");

    for (std::size_t i = 0; i < sequential_output.size(); ++i) {
        if (std::abs(sequential_output[i] - parallel_output[i]) > 1e-5f)
            throw std::runtime_error("OpenMP and sequential outputs differ.");
    }

    std::cout << "Sequential vs OpenMP correctness: PASS\n";

    benchmark_openmp(
        model,
        {1, 16, 64, 256, 1024},
        csv,
        warmups,
        trials
    );

#else

    std::cout << "Mini Inference Engine — Phase 1: Sequential CPU\n";

    correctness_checks(model);

    benchmark(
        model,
        {1, 16, 64, 256, 1024},
        csv,
        warmups,
        trials
    );

#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
