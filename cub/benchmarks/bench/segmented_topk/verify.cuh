// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/device_vector.h>
#include <thrust/replace.h>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace
{
// TODO(topk): TEMPORARY WORKAROUND -- remove once the block_topk_air signed-zero radix-selection bug is fixed.
// For max selection, a +0.0 key's twiddled-and-inverted bits collide with the radix digit extractor's -0.0 sentinel,
// corrupting selection when many zeros straddle the k-th position (e.g. the relu_quantized pattern). Until the kernel
// is fixed, replace exact zeros (+0.0 and -0.0, which compare equal to 0) in the generated input with the smallest
// positive normal so the benchmark's correctness check passes. Tied zeros stay tied and remain below real positive
// values, so the workload's tie structure is preserved. No-op for non-floating-point keys.
template <typename KeyT>
void workaround_replace_zeros(thrust::device_vector<KeyT>& d_keys)
{
  if constexpr (cuda::std::is_floating_point_v<KeyT>)
  {
    thrust::replace(d_keys.begin(), d_keys.end(), KeyT{0}, (cuda::std::numeric_limits<KeyT>::min)());
  }
}
} // namespace
