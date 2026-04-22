#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace sloth {
namespace operators {

template <typename T>
void add_cute_impl(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = 0);

} // namespace operators
} // namespace sloth