// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Explicit instantiations of the device top-k reference helpers for the closed native type set. Compiling the heavy
//! `DeviceSegmentedRadixSort` bodies once here keeps them out of every consuming test/benchmark translation unit.
//! Keep this list in sync with the `extern template` declarations in c2h/device_topk_reference.cuh.

#include <c2h/device_topk_reference.cuh>

namespace c2h
{

#define C2H_TOPK_REF_INSTANTIATE_KEY(KeyT)                                                         \
  template void segmented_sort_keys<KeyT>(                                                         \
    KeyT*,                                                                                         \
    cuda::std::int64_t,                                                                            \
    cuda::std::int64_t,                                                                            \
    const cuda::std::int64_t*,                                                                     \
    const cuda::std::int64_t*,                                                                     \
    cub::detail::topk::select);                                                                    \
  template bool verify_segmented_topk_keys<KeyT>(                                                  \
    const KeyT*,                                                                                   \
    cuda::std::int64_t,                                                                            \
    const KeyT*,                                                                                   \
    cuda::std::int64_t,                                                                            \
    cuda::std::int64_t,                                                                            \
    const cuda::std::int64_t*,                                                                     \
    cub::detail::topk::select)

#define C2H_TOPK_REF_INSTANTIATE_INDICES(KeyT, IndexT)                                             \
  template bool verify_segmented_topk_indices<KeyT, IndexT>(                                       \
    const KeyT*,                                                                                   \
    cuda::std::int64_t,                                                                            \
    const KeyT*,                                                                                   \
    const IndexT*,                                                                                 \
    cuda::std::int64_t,                                                                            \
    cuda::std::int64_t,                                                                            \
    const cuda::std::int64_t*)

C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::int8_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::int16_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::int32_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::int64_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::uint8_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::uint16_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::uint32_t);
C2H_TOPK_REF_INSTANTIATE_KEY(cuda::std::uint64_t);
C2H_TOPK_REF_INSTANTIATE_KEY(float);
C2H_TOPK_REF_INSTANTIATE_KEY(double);

C2H_TOPK_REF_INSTANTIATE_INDICES(float, cuda::std::int32_t);
C2H_TOPK_REF_INSTANTIATE_INDICES(float, cuda::std::int64_t);

#undef C2H_TOPK_REF_INSTANTIATE_KEY
#undef C2H_TOPK_REF_INSTANTIATE_INDICES

} // namespace c2h
