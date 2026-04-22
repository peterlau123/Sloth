#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace sloth {
namespace utils {

inline int get_cuda_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline void set_cuda_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

inline int get_current_cuda_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline size_t get_free_memory() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

inline size_t get_total_memory() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

struct CudaDeviceInfo {
    int device_id;
    char name[256];
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
};

inline CudaDeviceInfo get_cuda_device_info(int device = -1) {
    CudaDeviceInfo info;
    if (device < 0) {
        device = get_current_cuda_device();
    }
    info.device_id = device;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    strncpy(info.name, prop.name, 255);
    info.name[255] = '\0';
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    return info;
}

} // namespace utils
} // namespace sloth