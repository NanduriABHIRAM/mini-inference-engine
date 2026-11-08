#include "gpu_infer.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mie {
namespace {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    bool allocate(std::size_t count) {
        reset();
        if (count == 0) return false;
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) == cudaSuccess;
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

std::string cuda_error(cudaError_t status) {
    const char* message = cudaGetErrorString(status);
    return message != nullptr ? message : "unknown CUDA runtime error";
}

std::string cublas_error(cublasStatus_t status) {
    return "cuBLAS status " + std::to_string(static_cast<int>(status));
}

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

__global__ void sigmoid_kernel(float* z, int count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(count)) return;
    const float value = z[i];
    z[i] = value >= 0.0f ? 1.0f / (1.0f + expf(-value))
                         : expf(value) / (1.0f + expf(value));
}

}  // namespace

struct GpuInferenceSession::Impl {
    struct DeviceLayer {
        int in_features = 0;
        int out_features = 0;
        DeviceBuffer<float> weights;
        DeviceBuffer<float> bias;
    };

    int input_dim = 0;
    int output_dim = 0;
    int max_width = 0;
    OutputActivation output_activation = OutputActivation::Softmax;
    std::vector<DeviceLayer> layers;
    DeviceBuffer<float> input_buffer;
    DeviceBuffer<float> scratch_a;
    DeviceBuffer<float> scratch_b;
    std::size_t buffer_capacity = 0;
    cublasHandle_t handle = nullptr;
    bool is_ready = false;
    std::string last_error;

    explicit Impl(const MLP& net) { initialize(net); }

    ~Impl() {
        if (handle != nullptr) cublasDestroy(handle);
    }

    void set_error(const std::string& message) { last_error = message; }

    bool initialize(const MLP& net) {
        if (net.layers.empty() || net.input_dim() <= 0 || net.output_dim() <= 0 || net.max_width() <= 0) {
            set_error("model has invalid dimensions");
            return false;
        }
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            set_error("could not create a cuBLAS handle");
            return false;
        }

        input_dim = net.input_dim();
        output_dim = net.output_dim();
        max_width = net.max_width();
        output_activation = net.output_activation;
        layers.reserve(net.layers.size());
        int expected_features = input_dim;
        for (const DenseLayer& layer : net.layers) {
            if (layer.in_features != expected_features || layer.in_features <= 0 || layer.out_features <= 0 ||
                layer.W.size() != static_cast<std::size_t>(layer.in_features) * layer.out_features ||
                layer.b.size() != static_cast<std::size_t>(layer.out_features)) {
                set_error("model layers have incompatible dimensions");
                return false;
            }

            DeviceLayer uploaded;
            uploaded.in_features = layer.in_features;
            uploaded.out_features = layer.out_features;
            if (!uploaded.weights.allocate(layer.W.size())) {
                set_error("could not allocate GPU model weights: " + cuda_error(cudaGetLastError()));
                return false;
            }
            if (!uploaded.bias.allocate(layer.b.size())) {
                set_error("could not allocate GPU model bias: " + cuda_error(cudaGetLastError()));
                return false;
            }
            cudaError_t copy_status = cudaMemcpy(uploaded.weights.get(), layer.W.data(),
                                                  layer.W.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                set_error("could not upload GPU model weights: " + cuda_error(copy_status));
                return false;
            }
            copy_status = cudaMemcpy(uploaded.bias.get(), layer.b.data(),
                                     layer.b.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                set_error("could not upload GPU model bias: " + cuda_error(copy_status));
                return false;
            }
            layers.push_back(std::move(uploaded));
            expected_features = layer.out_features;
        }
        is_ready = true;
        return true;
    }

    bool ensure_buffers(int batch) {
        const std::size_t required = static_cast<std::size_t>(batch) * max_width;
        if (required <= buffer_capacity) return true;

    
        DeviceBuffer<float> new_input;
        DeviceBuffer<float> new_scratch_a;
        DeviceBuffer<float> new_scratch_b;
        if (!new_input.allocate(required) || !new_scratch_a.allocate(required) ||
            !new_scratch_b.allocate(required)) {
            set_error("could not allocate GPU inference buffers: " + cuda_error(cudaGetLastError()));
            return false;
        }
        input_buffer = std::move(new_input);
        scratch_a = std::move(new_scratch_a);
        scratch_b = std::move(new_scratch_b);
        buffer_capacity = required;
        return true;
    }

    bool forward(const float* x, float* y, int batch) {
        if (!is_ready || x == nullptr || y == nullptr || batch <= 0) {
            set_error("GPU session or inference input is invalid");
            return false;
        }
        if (!ensure_buffers(batch)) return false;

        cudaError_t copy_status = cudaMemcpy(input_buffer.get(), x,
                                             static_cast<std::size_t>(batch) * input_dim * sizeof(float),
                                             cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess) {
            set_error("could not upload inference input: " + cuda_error(copy_status));
            return false;
        }

        float* current = input_buffer.get();
        float* next = scratch_a.get();
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const DeviceLayer& layer = layers[layer_index];
            const int k = layer.in_features;
            const int n = layer.out_features;
            const bool is_last = layer_index + 1 == layers.size();

            // Row-major X[B,K] * W[N,K]^T is the transpose of W * X^T.
            // cuBLAS is column-major, so this writes directly into row-major
            // output memory [B,N].
            const cublasStatus_t gemm_status = cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, n, batch, k, &alpha, layer.weights.get(), k,
                current, k, &beta, next, n);
            if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                set_error("cuBLAS matrix multiply failed: " + cublas_error(gemm_status));
                return false;
            }

            constexpr int threads = 256;
            const std::size_t element_count = static_cast<std::size_t>(batch) * n;
            const int blocks = static_cast<int>((element_count + threads - 1) / threads);
            if (is_last) {
                bias_kernel<<<blocks, threads>>>(next, layer.bias.get(), batch, n);
                if (output_activation == OutputActivation::Softmax) {
                    const int row_blocks = (batch + threads - 1) / threads;
                    softmax_rows_kernel<<<row_blocks, threads>>>(next, batch, n);
                } else {
                    sigmoid_kernel<<<blocks, threads>>>(next, static_cast<int>(element_count));
                }
            } else {
                bias_relu_kernel<<<blocks, threads>>>(next, layer.bias.get(), batch, n);
            }
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                set_error("CUDA activation kernel failed: " + cuda_error(launch_status));
                return false;
            }
            current = next;
            next = next == scratch_a.get() ? scratch_b.get() : scratch_a.get();
        }

        copy_status = cudaMemcpy(y, current, static_cast<std::size_t>(batch) * output_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost);
        if (copy_status != cudaSuccess) {
            set_error("could not download inference output: " + cuda_error(copy_status));
            return false;
        }
        return true;
    }
};

GpuStatus query_gpu_status() {
    GpuStatus result;

    const CUresult driver_status = cuInit(0);
    if (driver_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(driver_status, &message);
        result.detail = message != nullptr ? message : "CUDA driver initialization failed";
        return result;
    }

    int device_count = 0;
    const cudaError_t runtime_status = cudaGetDeviceCount(&device_count);
    if (runtime_status != cudaSuccess) {
        result.detail = cudaGetErrorString(runtime_status);
        return result;
    }
    if (device_count == 0) {
        result.detail = "no CUDA-capable device was found";
        return result;
    }

    cudaDeviceProp properties {};
    const cudaError_t properties_status = cudaGetDeviceProperties(&properties, 0);
    if (properties_status != cudaSuccess) {
        result.detail = cudaGetErrorString(properties_status);
        return result;
    }

    result.available = true;
    result.device_count = device_count;
    result.detail = properties.name;
    return result;
}

GpuInferenceSession::GpuInferenceSession(const MLP& net) : impl_(std::make_unique<Impl>(net)) {}
GpuInferenceSession::~GpuInferenceSession() = default;
GpuInferenceSession::GpuInferenceSession(GpuInferenceSession&&) noexcept = default;
GpuInferenceSession& GpuInferenceSession::operator=(GpuInferenceSession&&) noexcept = default;

bool GpuInferenceSession::ready() const {
    return impl_ != nullptr && impl_->is_ready;
}

const std::string& GpuInferenceSession::error() const {
    static const std::string unavailable = "GPU session is unavailable";
    return impl_ != nullptr ? impl_->last_error : unavailable;
}

bool GpuInferenceSession::forward(const float* x, float* y, int batch) {
    return impl_ != nullptr && impl_->forward(x, y, batch);
}

bool forward_gpu(const MLP& net, const float* x, float* y, int batch) {
    GpuInferenceSession session(net);
    return session.ready() && session.forward(x, y, batch);
}

}  // namespace mie
