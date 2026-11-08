#include <cmath>
#include <vector>

#include "inference.hpp"

namespace mie {
namespace {

inline void relu(float* z, int n) {
    for (int i = 0; i < n; ++i) {
        z[i] = z[i] > 0.0f ? z[i] : 0.0f;
    }
}

void softmax_rows(float* z, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = z + static_cast<size_t>(r) * cols;
        float m = row[0];
        for (int j = 1; j < cols; ++j) {
            m = row[j] > m ? row[j] : m;
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - m);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < cols; ++j) {
            row[j] *= inv;
        }
    }
}

void sigmoid_rows(float* z, int rows, int cols) {
    for (int index = 0; index < rows * cols; ++index) {
        const float value = z[index];
        // This branch form avoids overflow for large-magnitude logits.
        z[index] = value >= 0.0f ? 1.0f / (1.0f + std::exp(-value))
                                 : std::exp(value) / (1.0f + std::exp(value));
    }
}

}  // namespace

bool forward_cpu(const MLP& net, const float* x, float* y, int batch) {
    if (net.layers.empty() || batch <= 0) return false;
    const int width = net.max_width();
    std::vector<float> buf_a(static_cast<size_t>(batch) * width);
    std::vector<float> buf_b(static_cast<size_t>(batch) * width);

    float* cur = buf_a.data();
    float* nxt = buf_b.data();
    std::copy(x, x + static_cast<size_t>(batch) * net.input_dim(), cur);

    for (size_t l = 0; l < net.layers.size(); ++l) {
        const DenseLayer& layer = net.layers[l];
        const int K = layer.in_features;
        const int N = layer.out_features;

        for (int m = 0; m < batch; ++m) {
            const float* a = cur + static_cast<size_t>(m) * K;
            float* z = nxt + static_cast<size_t>(m) * N;

            for (int n = 0; n < N; ++n) {
                // W is [N][K], so this row is exactly W[n].
                const float* w = layer.W.data() + static_cast<size_t>(n) * K;
                float acc = layer.b[n];
                for (int k = 0; k < K; ++k) {
                    acc += a[k] * w[k];
                }
                z[n] = acc;
            }
        }

        const bool is_last = (l + 1 == net.layers.size());
        if (is_last) {
            if (net.output_activation == OutputActivation::Softmax) {
                softmax_rows(nxt, batch, N);
            } else {
                sigmoid_rows(nxt, batch, N);
            }
        } else {
            relu(nxt, batch * N);
        }
        std::swap(cur, nxt);
    }

    std::copy(cur, cur + static_cast<size_t>(batch) * net.output_dim(), y);
    return true;
}

#ifdef _OPENMP
bool forward_cpu_omp(const MLP& net, const float* x, float* y, int batch) {
    if (net.layers.empty() || batch <= 0) return false;

    const int width = net.max_width();
    std::vector<float> buf_a(static_cast<size_t>(batch) * width);
    std::vector<float> buf_b(static_cast<size_t>(batch) * width);

    float* cur = buf_a.data();
    float* nxt = buf_b.data();
    std::copy(x, x + static_cast<size_t>(batch) * net.input_dim(), cur);

    for (size_t l = 0; l < net.layers.size(); ++l) {
        const DenseLayer& layer = net.layers[l];
        const int K = layer.in_features;
        const int N = layer.out_features;

        // Two independent parallel regions, one per elementwise op, so each
        // gets a clean barrier. `schedule(static)` hands each thread a fixed
        // contiguous block up front instead of a dynamic queue: the per-sample
        // work is identical for every m, the cost is identical, so there is
        // nothing to balance at runtime and the static split has no overhead.
#pragma omp parallel for schedule(static)
        for (int m = 0; m < batch; ++m) {
            const float* a = cur + static_cast<size_t>(m) * K;
            float* z = nxt + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                const float* w = layer.W.data() + static_cast<size_t>(n) * K;
                float acc = layer.b[n];
                for (int k = 0; k < K; ++k) {
                    acc += a[k] * w[k];
                }
                z[n] = acc;
            }
        }

        const bool is_last = (l + 1 == net.layers.size());
        if (is_last) {
            // Both output activations are row-local, so they parallelise over rows.
#pragma omp parallel for schedule(static)
            for (int m = 0; m < batch; ++m) {
                float* row = nxt + static_cast<size_t>(m) * N;
                if (net.output_activation == OutputActivation::Softmax) {
                    softmax_rows(row, 1, N);
                } else {
                    sigmoid_rows(row, 1, N);
                }
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < batch * N; ++i) {
                nxt[i] = nxt[i] > 0.0f ? nxt[i] : 0.0f;
            }
        }
        std::swap(cur, nxt);
    }

    std::copy(cur, cur + static_cast<size_t>(batch) * net.output_dim(), y);
    return true;
}
#endif

const char* backend_name_cpu() { return "CPU (1 thread)"; }
const char* backend_name_omp() { return "CPU (OpenMP)"; }
const char* backend_name_gpu() { return "GPU (CUDA+cuBLAS)"; }

DiffStats compare_outputs(const float* a, const float* b, size_t n) {
    DiffStats s;
    if (n == 0) return s;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += d * d;
        const double ad = d < 0 ? -d : d;
        if (ad > s.max_abs) s.max_abs = ad;
    }
    s.rms = std::sqrt(sum_sq / static_cast<double>(n));
    return s;
}

}  // namespace mie
