#pragma once

#include "inference.hpp"

#include <memory>
#include <string>

namespace mie {

struct GpuStatus {
    bool available = false;
    int device_count = 0;
    std::string detail;
};

// Checks that the CUDA driver and runtime can see at least one usable device.
GpuStatus query_gpu_status();

// Keeps a cuBLAS handle, model parameters, and reusable device scratch buffers
// alive across calls.  Use one session for a long-lived inference process so
// GPU timing reflects inference rather than recreating a context and uploading
// immutable weights for every request.
class GpuInferenceSession {
public:
    explicit GpuInferenceSession(const MLP& net);
    ~GpuInferenceSession();

    GpuInferenceSession(const GpuInferenceSession&) = delete;
    GpuInferenceSession& operator=(const GpuInferenceSession&) = delete;
    GpuInferenceSession(GpuInferenceSession&&) noexcept;
    GpuInferenceSession& operator=(GpuInferenceSession&&) noexcept;

    bool ready() const;
    const std::string& error() const;
    bool forward(const float* x, float* y, int batch);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs one batched MLP forward pass with cuBLAS dense layers and CUDA
// activation/softmax kernels. This compatibility helper deliberately creates a
// temporary session; callers making repeated predictions should use
// GpuInferenceSession instead.
bool forward_gpu(const MLP& net, const float* x, float* y, int batch);

}  // namespace mie
