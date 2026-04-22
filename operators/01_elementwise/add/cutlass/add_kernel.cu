#include <cuda.h>
#include <cuda_runtime.h>

namespace sloth {
namespace operators {

template <typename T>
__global__ void add_cutlass_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
void add_cutlass_impl(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_cutlass_kernel<T><<<grid_size, block_size, 0, stream>>>(a, b, c, n);
}

template void add_cutlass_impl<float>(const float*, const float*, float*, size_t, cudaStream_t);
template void add_cutlass_impl<double>(const double*, const double*, double*, size_t, cudaStream_t);
template void add_cutlass_impl<half>(const half*, const half*, half*, size_t, cudaStream_t);

} // namespace operators
} // namespace sloth