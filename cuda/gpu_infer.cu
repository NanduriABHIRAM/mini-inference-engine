#include "gpu_infer.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>

namespace mie {
namespace {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    bool allocate(std::size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) == cudaSuccess;
    }

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

__global__ void bias_relu_kernel(float* z, const float* bias, int rows, int cols) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    if (i < count) {
        const int col = static_cast<int>(i % cols);
        const float value = z[i] + bias[col];
        z[i] = value > 0.0f ? value : 0.0f;
    }
}

__global__ void bias_kernel(float* z, const float* bias, int rows, int cols) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    if (i < count) z[i] += bias[i % cols];
}

// One CUDA thread handles one output row. This keeps the implementation
// simple and stable for the small classifier output dimension.
__global__ void softmax_rows_kernel(float* z, int rows, int cols) {
    const int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= rows) return;

    float* row = z + static_cast<std::size_t>(row_index) * cols;
    float maximum = row[0];
    for (int j = 1; j < cols; ++j) {
        if (row[j] > maximum) maximum = row[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        row[j] = expf(row[j] - maximum);
        sum += row[j];
    }
    const float inverse_sum = 1.0f / sum;
    for (int j = 0; j < cols; ++j) row[j] *= inverse_sum;
}

}  // namespace

bool forward_gpu(const MLP& net, const float* x, float* y, int batch) {
    if (net.layers.empty() || batch <= 0 || x == nullptr || y == nullptr) return false;
    const int input_dim = net.input_dim();
    const int output_dim = net.output_dim();
    const int width = net.max_width();
    if (input_dim <= 0 || output_dim <= 0 || width <= 0) return false;

    const std::size_t buffer_count = static_cast<std::size_t>(batch) * width;
    DeviceBuffer<float> input_buffer;
    DeviceBuffer<float> scratch_a;
    DeviceBuffer<float> scratch_b;
    if (!input_buffer.allocate(buffer_count) || !scratch_a.allocate(buffer_count) ||
        !scratch_b.allocate(buffer_count)) return false;
    if (cudaMemcpy(input_buffer.get(), x,
                   static_cast<std::size_t>(batch) * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return false;

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;

    float* current = input_buffer.get();
    float* next = scratch_a.get();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int expected_features = input_dim;
    bool success = true;

    for (std::size_t layer_index = 0; layer_index < net.layers.size(); ++layer_index) {
        const DenseLayer& layer = net.layers[layer_index];
        const int k = layer.in_features;
        const int n = layer.out_features;
        const bool is_last = layer_index + 1 == net.layers.size();
        if (k != expected_features || k <= 0 || n <= 0 ||
            layer.W.size() != static_cast<std::size_t>(k) * n ||
            layer.b.size() != static_cast<std::size_t>(n)) {
            success = false;
            break;
        }

        DeviceBuffer<float> weights;
        DeviceBuffer<float> bias;
        if (!weights.allocate(layer.W.size()) || !bias.allocate(layer.b.size()) ||
            cudaMemcpy(weights.get(), layer.W.data(), layer.W.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(bias.get(), layer.b.data(), layer.b.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            success = false;
            break;
        }

        // Row-major X[B,K] * W[N,K]^T is the transpose of W * X^T.
        // cuBLAS uses column-major matrices, so that form writes directly into
        // the same memory layout used for the row-major output [B,N].
        if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, batch, k, &alpha, weights.get(), k,
                        current, k, &beta, next, n) != CUBLAS_STATUS_SUCCESS) {
            success = false;
            break;
        }

        constexpr int threads = 256;
        const std::size_t element_count = static_cast<std::size_t>(batch) * n;
        const int blocks = static_cast<int>((element_count + threads - 1) / threads);
        if (is_last) {
            bias_kernel<<<blocks, threads>>>(next, bias.get(), batch, n);
            if (cudaGetLastError() != cudaSuccess) {
                success = false;
                break;
            }
            const int row_blocks = (batch + threads - 1) / threads;
            softmax_rows_kernel<<<row_blocks, threads>>>(next, batch, n);
        } else {
            bias_relu_kernel<<<blocks, threads>>>(next, bias.get(), batch, n);
        }
        if (cudaGetLastError() != cudaSuccess) {
            success = false;
            break;
        }
        current = next;
        next = next == scratch_a.get() ? scratch_b.get() : scratch_a.get();
        expected_features = n;
    }

    if (success && cudaMemcpy(y, current,
                             static_cast<std::size_t>(batch) * output_dim * sizeof(float),
                             cudaMemcpyDeviceToHost) != cudaSuccess) success = false;
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) success = false;
    return success;
}

}  // namespace mie
