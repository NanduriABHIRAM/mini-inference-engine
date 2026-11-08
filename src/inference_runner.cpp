#include "inference.hpp"

#ifdef MIE_HAS_CUDA
#include "gpu_infer.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

enum class Backend { Cpu, OpenMP, Gpu };

struct Options {
    std::string backend = "auto";
    fs::path input_path;
    fs::path output_path;
    fs::path model_path;
    int samples = 1;
    int scheduler_warmups = 1;
    int scheduler_trials = 5;
    float decision_threshold = 0.5f;
    bool has_input = false;
    bool has_model = false;
    bool samples_was_set = false;
};

struct SchedulerTimings {
    double cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double openmp_ms = std::numeric_limits<double>::quiet_NaN();
    double gpu_ms = std::numeric_limits<double>::quiet_NaN();
    double cpu_openmp_max_abs_difference = std::numeric_limits<double>::quiet_NaN();
    double cpu_gpu_max_abs_difference = std::numeric_limits<double>::quiet_NaN();
};

struct Candidate {
    Backend backend = Backend::Cpu;
    bool valid = false;
    double median_ms = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> output;
    std::string reason;
};

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char character) {
        return std::isspace(character) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string lower(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> values;
    std::size_t start = 0;
    while (start <= line.size()) {
        const std::size_t comma = line.find(',', start);
        values.push_back(trim(line.substr(start, comma - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

bool parse_finite_float(const std::string& token, float* value) {
    try {
        std::size_t position = 0;
        const float parsed = std::stof(token, &position);
        if (position != token.size() || !std::isfinite(parsed)) return false;
        *value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<float> load_feature_csv(const fs::path& path, int feature_count) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open input CSV: " + path.string());

    std::vector<float> features;
    std::string line;
    int line_number = 0;
    bool saw_first_row = false;

    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty()) continue;

        const std::vector<std::string> cells = split_csv_row(line);
        std::vector<float> row;
        row.reserve(cells.size());
        bool numeric = true;
        bool has_numeric_cell = false;
        for (const std::string& cell : cells) {
            float value = 0.0f;
            if (!parse_finite_float(cell, &value)) {
                numeric = false;
                continue;
            }
            has_numeric_cell = true;
            row.push_back(value);
        }

        if (!numeric) {
            if (!saw_first_row && !has_numeric_cell) {
                if (static_cast<int>(cells.size()) != feature_count) {
                    throw std::runtime_error("CSV header must contain exactly " +
                                             std::to_string(feature_count) + " columns.");
                }
                saw_first_row = true;
                continue;
            }
            throw std::runtime_error("CSV line " + std::to_string(line_number) +
                                     " contains a non-numeric feature value.");
        }

        saw_first_row = true;
        if (static_cast<int>(row.size()) != feature_count) {
            throw std::runtime_error("CSV line " + std::to_string(line_number) + " has " +
                                     std::to_string(row.size()) + " columns; expected " +
                                     std::to_string(feature_count) + ".");
        }
        features.insert(features.end(), row.begin(), row.end());
    }

    if (features.empty()) throw std::runtime_error("Input CSV contains no feature rows.");
    return features;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--backend" && i + 1 < argc) {
            options.backend = lower(argv[++i]);
        } else if (argument == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
            options.has_input = true;
        } else if (argument == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (argument == "--model" && i + 1 < argc) {
            options.model_path = argv[++i];
            options.has_model = true;
        } else if (argument == "--samples" && i + 1 < argc) {
            options.samples = std::stoi(argv[++i]);
            options.samples_was_set = true;
        } else if (argument == "--scheduler-warmups" && i + 1 < argc) {
            options.scheduler_warmups = std::stoi(argv[++i]);
        } else if (argument == "--scheduler-trials" && i + 1 < argc) {
            options.scheduler_trials = std::stoi(argv[++i]);
        } else if (argument == "--decision-threshold" && i + 1 < argc) {
            options.decision_threshold = std::stof(argv[++i]);
        } else if (argument == "--help") {
            std::cout << "Usage: inference_runner --backend cpu|openmp|gpu|auto --output PATH "
                         "[--model MODEL.mie] [--input FEATURES.csv | --samples N] "
                         "[--scheduler-warmups N --scheduler-trials N] "
                         "[--decision-threshold VALUE]\n"
                         "Without --model, the deterministic 512-feature benchmark model is used. "
                         "With --model, CSV rows must match the model input size.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }

    if (options.output_path.empty()) throw std::runtime_error("--output PATH is required.");
    if (options.has_input && options.samples_was_set)
        throw std::runtime_error("Use either --input or --samples, not both.");
    if (!options.has_input && options.samples <= 0)
        throw std::runtime_error("--samples must be greater than zero.");
    if (options.has_model && !options.has_input)
        throw std::runtime_error("A custom --model requires --input FEATURES.csv.");
    if (options.scheduler_warmups < 0 || options.scheduler_trials < 1)
        throw std::runtime_error("scheduler warmups must be >= 0 and trials must be >= 1.");
    if (!std::isfinite(options.decision_threshold) || options.decision_threshold < 0.0f ||
        options.decision_threshold > 1.0f) {
        throw std::runtime_error("--decision-threshold must be between 0 and 1.");
    }
    if (options.backend == "sequential_cpu") options.backend = "cpu";
    if (options.backend == "openmp_cpu") options.backend = "openmp";
    if (options.backend == "cuda_gpu") options.backend = "gpu";
    if (options.backend != "cpu" && options.backend != "openmp" &&
        options.backend != "gpu" && options.backend != "auto") {
        throw std::runtime_error("--backend must be cpu, openmp, gpu, or auto.");
    }
    return options;
}

const char* backend_id(Backend backend) {
    switch (backend) {
        case Backend::Cpu: return "sequential_cpu";
        case Backend::OpenMP: return "openmp_cpu";
        case Backend::Gpu: return "cuda_gpu";
    }
    return "unknown";
}

bool validate_probabilities(const std::vector<float>& values, int batch, const mie::MLP& model) {
    const int classes = model.output_dim();
    if (values.size() != static_cast<std::size_t>(batch) * classes) return false;
    for (int row = 0; row < batch; ++row) {
        double sum = 0.0;
        for (int column = 0; column < classes; ++column) {
            const float value = values[static_cast<std::size_t>(row) * classes + column];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
            sum += value;
        }
        if (model.output_activation == mie::OutputActivation::Softmax && std::abs(sum - 1.0) > 1e-5)
            return false;
    }
    return true;
}

bool run_backend(Backend backend, const mie::MLP& model, const std::vector<float>& input,
                 std::vector<float>& output, int batch) {
    switch (backend) {
        case Backend::Cpu:
            return mie::forward_cpu(model, input.data(), output.data(), batch);
        case Backend::OpenMP:
#ifdef _OPENMP
            return mie::forward_cpu_omp(model, input.data(), output.data(), batch);
#else
            return false;
#endif
        case Backend::Gpu:
#ifdef MIE_HAS_CUDA
            return mie::forward_gpu(model, input.data(), output.data(), batch);
#else
            return false;
#endif
    }
    return false;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

Candidate profile_backend(Backend backend, const mie::MLP& model, const std::vector<float>& input,
                          int batch, int warmups, int trials) {
    Candidate result;
    result.backend = backend;
    result.output.resize(static_cast<std::size_t>(batch) * model.output_dim());
    for (int attempt = 0; attempt < warmups; ++attempt) {
        if (!run_backend(backend, model, input, result.output, batch)) {
            result.reason = "backend execution failed during warmup";
            return result;
        }
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(trials));
    for (int attempt = 0; attempt < trials; ++attempt) {
        const auto start = Clock::now();
        if (!run_backend(backend, model, input, result.output, batch)) {
            result.reason = "backend execution failed during measurement";
            return result;
        }
        timings.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    if (!validate_probabilities(result.output, batch, model)) {
        result.reason = "backend returned invalid probability values";
        return result;
    }
    result.median_ms = median(std::move(timings));
    result.valid = true;
    return result;
}

void record_scheduler_timing(SchedulerTimings* timings, const Candidate& candidate) {
    if (!candidate.valid) return;
    switch (candidate.backend) {
        case Backend::Cpu: timings->cpu_ms = candidate.median_ms; break;
        case Backend::OpenMP: timings->openmp_ms = candidate.median_ms; break;
        case Backend::Gpu: timings->gpu_ms = candidate.median_ms; break;
    }
}

void write_optional_number(std::ofstream& file, double value) {
    if (std::isfinite(value)) file << value;
}

void write_results(const fs::path& path, const std::string& requested_backend,
                   const std::string& selected_backend, const std::string& input_source,
                   double inference_latency_ms, double scheduler_selection_latency_ms,
                   const mie::MLP& model, float decision_threshold,
                   const SchedulerTimings& scheduler, const std::vector<float>& output, int batch) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot open output CSV: " + path.string());

    const int classes = model.output_dim();
    const double total_request_latency_ms = std::isfinite(scheduler_selection_latency_ms)
                                                ? scheduler_selection_latency_ms + inference_latency_ms
                                                : inference_latency_ms;
    file << "requested_backend,selected_backend,input_source,latency_ms,"
            "scheduler_selection_latency_ms,total_request_latency_ms,sample_index,"
            "predicted_class,predicted_probability,fraud_probability,output_activation,"
            "decision_threshold,scheduler_cpu_median_ms,scheduler_openmp_median_ms,"
            "scheduler_gpu_median_ms,scheduler_cpu_openmp_max_abs_difference,"
            "scheduler_cpu_gpu_max_abs_difference";
    for (int column = 0; column < classes; ++column) file << ",class_" << column << "_probability";
    file << '\n';

    file << std::fixed << std::setprecision(9);
    for (int row = 0; row < batch; ++row) {
        const float* probabilities = output.data() + static_cast<std::size_t>(row) * classes;
        int predicted_class = 0;
        float predicted_probability = 0.0f;
        float fraud_probability = std::numeric_limits<float>::quiet_NaN();
        if (model.output_activation == mie::OutputActivation::Sigmoid) {
            fraud_probability = probabilities[0];
            predicted_class = fraud_probability >= decision_threshold ? 1 : 0;
            predicted_probability = predicted_class == 1 ? fraud_probability : 1.0f - fraud_probability;
        } else {
            const float* best = std::max_element(probabilities, probabilities + classes);
            predicted_class = static_cast<int>(best - probabilities);
            predicted_probability = *best;
        }
        file << requested_backend << ',' << selected_backend << ',' << input_source << ','
             << inference_latency_ms << ',';
        write_optional_number(file, scheduler_selection_latency_ms);
        file << ',' << total_request_latency_ms << ',' << row << ',' << predicted_class << ','
             << predicted_probability << ',';
        if (std::isfinite(fraud_probability)) file << fraud_probability;
        file << ',' << (model.output_activation == mie::OutputActivation::Sigmoid ? "sigmoid" : "softmax")
             << ',' << decision_threshold << ',';
        write_optional_number(file, scheduler.cpu_ms);
        file << ',';
        write_optional_number(file, scheduler.openmp_ms);
        file << ',';
        write_optional_number(file, scheduler.gpu_ms);
        file << ',';
        write_optional_number(file, scheduler.cpu_openmp_max_abs_difference);
        file << ',';
        write_optional_number(file, scheduler.cpu_gpu_max_abs_difference);
        for (int column = 0; column < classes; ++column) file << ',' << probabilities[column];
        file << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const mie::MLP model = options.has_model ? mie::load_mlp_file(options.model_path.string())
                                                  : mie::make_reference_model();
        const int features = model.input_dim();
        if (features <= 0 || model.output_dim() <= 0)
            throw std::runtime_error("Model has invalid input or output dimensions.");

        bool gpu_available = false;
        std::string gpu_detail = "CUDA support was not built into this executable.";
#ifdef MIE_HAS_CUDA
        const mie::GpuStatus gpu_status = mie::query_gpu_status();
        gpu_available = gpu_status.available;
        gpu_detail = gpu_status.detail;
#endif
        // A fraud prediction must remain available when CUDA is missing (or
        // temporarily inaccessible, such as in a restricted WSL session).
        // Preserve the user's requested backend in the output, but use the
        // same measured adaptive path as --backend auto to select a verified
        // CPU/OpenMP fallback.
        const bool falling_back_from_gpu = options.backend == "gpu" && !gpu_available;
        const std::string execution_backend = falling_back_from_gpu ? "auto" : options.backend;

        std::vector<float> input;
        std::string input_source;
        if (options.has_input) {
            input = load_feature_csv(options.input_path, features);
            input_source = "uploaded_csv";
        } else {
            input = mie::make_reference_input(options.samples, features);
            input_source = "synthetic";
        }
        const int batch = static_cast<int>(input.size() / static_cast<std::size_t>(features));
        if (batch <= 0) throw std::runtime_error("Input does not contain a complete feature row.");

        SchedulerTimings scheduler;
        Backend selected = Backend::Cpu;
        double scheduler_selection_latency_ms = std::numeric_limits<double>::quiet_NaN();
        if (execution_backend == "auto") {
            const auto scheduler_start = Clock::now();
            std::vector<Candidate> candidates;
            candidates.reserve(3);
            candidates.push_back(profile_backend(Backend::Cpu, model, input, batch,
                                                 options.scheduler_warmups, options.scheduler_trials));
            Candidate& cpu_candidate = candidates.back();
            if (!cpu_candidate.valid)
                throw std::runtime_error("CPU scheduling reference failed: " + cpu_candidate.reason);
            record_scheduler_timing(&scheduler, cpu_candidate);

#ifdef _OPENMP
            candidates.push_back(profile_backend(Backend::OpenMP, model, input, batch,
                                                 options.scheduler_warmups, options.scheduler_trials));
            Candidate& openmp_candidate = candidates.back();
            if (openmp_candidate.valid) {
                const mie::DiffStats difference = mie::compare_outputs(
                    cpu_candidate.output.data(), openmp_candidate.output.data(), cpu_candidate.output.size());
                scheduler.cpu_openmp_max_abs_difference = difference.max_abs;
                if (difference.max_abs > 1e-5) {
                    openmp_candidate.valid = false;
                    openmp_candidate.reason = "CPU comparison exceeded 1e-5";
                }
            }
            record_scheduler_timing(&scheduler, openmp_candidate);
#endif

#ifdef MIE_HAS_CUDA
            if (gpu_available) {
                candidates.push_back(profile_backend(Backend::Gpu, model, input, batch,
                                                     options.scheduler_warmups, options.scheduler_trials));
                Candidate& gpu_candidate = candidates.back();
                if (gpu_candidate.valid) {
                    const mie::DiffStats difference = mie::compare_outputs(
                        cpu_candidate.output.data(), gpu_candidate.output.data(), cpu_candidate.output.size());
                    scheduler.cpu_gpu_max_abs_difference = difference.max_abs;
                    if (difference.max_abs > 1e-4) {
                        gpu_candidate.valid = false;
                        gpu_candidate.reason = "CPU comparison exceeded 1e-4";
                    }
                }
                record_scheduler_timing(&scheduler, gpu_candidate);
            }
#endif

            const Candidate* fastest = nullptr;
            for (const Candidate& candidate : candidates) {
                if (candidate.valid && (fastest == nullptr || candidate.median_ms < fastest->median_ms))
                    fastest = &candidate;
                std::cout << "Scheduler " << backend_id(candidate.backend) << ": ";
                if (candidate.valid) {
                    std::cout << std::fixed << std::setprecision(4) << candidate.median_ms << " ms\n";
                } else {
                    std::cout << "excluded (" << candidate.reason << ")\n";
                }
            }
            if (fastest == nullptr) throw std::runtime_error("No backend passed adaptive scheduling checks.");
            selected = fastest->backend;
            scheduler_selection_latency_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - scheduler_start).count();
        } else if (execution_backend == "cpu") {
            selected = Backend::Cpu;
        } else if (execution_backend == "openmp") {
            selected = Backend::OpenMP;
        } else {
            selected = Backend::Gpu;
        }

        std::vector<float> output(static_cast<std::size_t>(batch) * model.output_dim());
        const auto start = Clock::now();
        const bool success = run_backend(selected, model, input, output, batch);
        const double latency_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (!success) throw std::runtime_error("The selected backend returned an execution error.");
        if (!validate_probabilities(output, batch, model))
            throw std::runtime_error("The selected backend produced invalid probability output.");

        write_results(options.output_path, options.backend, backend_id(selected), input_source, latency_ms,
                      scheduler_selection_latency_ms, model, options.decision_threshold, scheduler, output,
                      batch);
        std::cout << "Selected backend: " << backend_id(selected) << '\n'
                  << "Input source: " << input_source << '\n'
                  << "Samples: " << batch << '\n'
                  << "Latency: " << std::fixed << std::setprecision(3) << latency_ms << " ms\n";
        if (execution_backend == "auto") {
            std::cout << "CUDA availability: " << (gpu_available ? "available" : "unavailable")
                      << " (" << gpu_detail << ")\n"
                      << "Scheduler selection overhead: " << std::fixed << std::setprecision(3)
                      << scheduler_selection_latency_ms << " ms\n"
                      << "Total request latency: " << scheduler_selection_latency_ms + latency_ms << " ms\n";
        }
        if (falling_back_from_gpu)
            std::cout << "CUDA was requested but unavailable; selected the fastest verified fallback.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
