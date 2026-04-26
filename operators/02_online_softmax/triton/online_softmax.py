"""
Online Softmax Implementation using Triton.

This module implements the online softmax algorithm using Triton GPU kernels.
Online softmax computes the softmax in a streaming fashion, processing data
in chunks while maintaining running max and sum values for numerical stability.

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/abs/2205.14135

Copyright (c) 2026 peterlau123
Licensed under the MIT License.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def online_softmax_kernel(x_ptr,stride_elements,output_ptr,STEP_SIZE:tl.constexpr,BLOCK_SIZE: tl.constexpr):
    pid= tl.program_id(0)
    block_start = pid * BLOCK_SIZE*stride_elements
    row_ptrs = block_start + tl.range(0, BLOCK_SIZE)*stride_elements
    # for each block, compute the max and sum in an online manner
    step_num_elements=stride_elements//STEP_SIZE
    maxs=tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    sums=tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    for k in tl.range(STEP_SIZE):
        offsets = row_ptrs+k*step_num_elements+tl.arange(0, step_num_elements)
        mask=offsets-row_ptrs<stride_elements
        data = tl.load(offsets,mask=mask)
        new_maxs = tl.maximum(maxs, data)
        new_sums = tl.exp(data - new_maxs) * mask + tl.exp(maxs - new_maxs) * sums
        step_output = tl.cdiv(tl.exp(data-new_maxs),new_sums)
        tl.store(offsets, step_output, mask=mask)
        maxs = new_maxs
        sums = new_sums
def online_softmax(x):
    assert x.device == DEVICE, f"Input tensor must be on device {DEVICE}, but got {x.device}"
    output = torch.empty_like(x)
    def grid(BLOCK_SIZE):
        return x.shape[0] // BLOCK_SIZE
    online_softmax_kernel[grid](x, x.stride(0), output, STEP_SIZE=4,BLOCK_SIZE=4)
    return output
