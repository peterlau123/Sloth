#pragma once

#include <chrono>
#include <cuda_runtime.h>

namespace sloth {
namespace utils {

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        stream_ = stream;
        cudaEventRecord(start_, stream_);
    }

    void stop() {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
    }

    float elapsed_milliseconds() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    float elapsed_seconds() const {
        return elapsed_milliseconds() / 1000.0f;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_;
};

class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_milliseconds() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(stop_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;
};

} // namespace utils
} // namespace sloth