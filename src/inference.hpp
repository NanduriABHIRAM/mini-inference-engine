#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mie {

// Dense layer weights are stored row-major as [out_features][in_features].
struct DenseLayer {
    int in_features = 0;
    int out_features = 0;
    std::vector<float> W;
    std::vector<float> b;
};

struct MLP {
    std::vector<DenseLayer> layers;

    int input_dim() const {
        return layers.empty() ? 0 : layers.front().in_features;
    }

    int output_dim() const {
        return layers.empty() ? 0 : layers.back().out_features;
    }

    int max_width() const {
        int width = input_dim();
        for (const DenseLayer& layer : layers) {
            width = std::max(width, layer.out_features);
        }
        return width;
    }
};

struct DiffStats {
    double max_abs = 0.0;
    double rms = 0.0;
};

// Shared deterministic workload used by every backend and benchmark runner.
MLP make_reference_model();
std::vector<float> make_reference_input(int batch, int features);

bool forward_cpu(const MLP& net, const float* x, float* y, int batch);
#ifdef _OPENMP
bool forward_cpu_omp(const MLP& net, const float* x, float* y, int batch);
#endif
bool forward_gpu(const MLP& net, const float* x, float* y, int batch);

const char* backend_name_cpu();
const char* backend_name_omp();
const char* backend_name_gpu();
DiffStats compare_outputs(const float* a, const float* b, std::size_t n);

}  // namespace mie
