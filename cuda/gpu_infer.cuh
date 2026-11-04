#pragma once

#include "inference.hpp"

namespace mie {

// Runs one batched MLP forward pass with cuBLAS dense layers and CUDA
// activation/softmax kernels. Returns false if a CUDA/cuBLAS operation fails.
bool forward_gpu(const MLP& net, const float* x, float* y, int batch);

}  // namespace mie
