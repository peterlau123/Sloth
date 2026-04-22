#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

namespace sloth {
namespace utils {

template <typename T>
bool allclose(const T* a, const T* b, size_t n, T rtol = static_cast<T>(1e-5), T atol = static_cast<T>(1e-8)) {
    for (size_t i = 0; i < n; ++i) {
        T diff = std::abs(a[i] - b[i]);
        T threshold = atol + rtol * std::abs(b[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

template <typename T>
T max_abs_error(const T* a, const T* b, size_t n) {
    T max_err = 0;
    for (size_t i = 0; i < n; ++i) {
        T err = std::abs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

template <typename T>
T mean_abs_error(const T* a, const T* b, size_t n) {
    T sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / static_cast<T>(n);
}

inline bool check_tensor(const torch::Tensor& actual, const torch::Tensor& expected,
                         double rtol = 1e-5, double atol = 1e-8) {
    return torch::allclose(actual, expected, rtol, atol).item<bool>();
}

inline double max_abs_error_tensor(const torch::Tensor& a, const torch::Tensor& b) {
    return (a - b).abs().max().item<double>();
}

inline double mean_abs_error_tensor(const torch::Tensor& a, const torch::Tensor& b) {
    return (a - b).abs().mean().item<double>();
}

struct CheckResult {
    bool passed;
    double max_error;
    double mean_error;
    std::string message;
};

inline CheckResult check_result(const torch::Tensor& actual, const torch::Tensor& expected,
                                double rtol = 1e-5, double atol = 1e-8) {
    CheckResult result;
    result.passed = check_tensor(actual, expected, rtol, atol);
    result.max_error = max_abs_error_tensor(actual, expected);
    result.mean_error = mean_abs_error_tensor(actual, expected);

    if (result.passed) {
        result.message = "PASSED";
    } else {
        result.message = "FAILED: max_error=" + std::to_string(result.max_error) +
                        ", mean_error=" + std::to_string(result.mean_error);
    }
    return result;
}

} // namespace utils
} // namespace sloth