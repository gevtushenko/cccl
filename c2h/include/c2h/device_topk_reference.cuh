// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Catch2-free, sort-based reference implementation and verifiers for (batched/segmented) device top-k.
//!
//! Shared by both the CUB tests and the CUB benchmarks. The expensive `DeviceSegmentedRadixSort` instantiations are
//! compiled once into the `cccl.c2h.core` static library for the closed native type set declared `extern template`
//! below; non-native key types (e.g. the c2h `half_t`/`bfloat16_t` wrappers or custom test keys) still work via the
//! in-header template definitions, instantiated in the consuming translation unit.
//!
//! The public boundary is intentionally concrete -- raw pointers plus `int64` segment offsets -- so the heavy sort is
//! instantiated only over the key/index type, never over an iterator type. Callers materialize their offsets first.

#pragma once

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh> // cub::detail::topk::select
#include <cub/util_type.cuh> // cub::DoubleBuffer

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>

namespace c2h
{

//! Sorts each segment `[d_begin_offsets[i], d_end_offsets[i])` of `d_keys` in place, in the given direction.
//!
//! Uses `cub::DeviceSegmentedRadixSort` (rather than `cub::DeviceSegmentedSort`): for a reference that only needs
//! per-segment ordering of arithmetic keys it compiles substantially faster at negligible runtime cost.
template <typename KeyT>
void segmented_sort_keys(
  KeyT* d_keys,
  cuda::std::int64_t num_items,
  cuda::std::int64_t num_segments,
  const cuda::std::int64_t* d_begin_offsets,
  const cuda::std::int64_t* d_end_offsets,
  cub::detail::topk::select direction);

//! Verifies the top-k key output against a segmented-sort reference.
//!   d_in_keys  : full input buffer; segment i live region = [i * in_stride, i * in_stride + seg_sizes[i])
//!   d_out_keys : output buffer of num_segments * k items; segment i = [i * k, (i + 1) * k)
//! Assumes every segment has at least k live items, so each output segment contains exactly k items.
template <typename KeyT>
[[nodiscard]] bool verify_segmented_topk_keys(
  const KeyT* d_in_keys,
  cuda::std::int64_t in_stride,
  const KeyT* d_out_keys,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t k,
  const cuda::std::int64_t* d_seg_sizes,
  cub::detail::topk::select direction);

//! Verifies the index (arg-top-k) output. Output indices are segment-local:
//!   - every reported index lies in [0, seg_sizes[seg]) for its segment,
//!   - the input key at that index equals the reported output key, and
//!   - indices are unique within each output segment.
template <typename KeyT, typename IndexT>
[[nodiscard]] bool verify_segmented_topk_indices(
  const KeyT* d_in_keys,
  cuda::std::int64_t in_stride,
  const KeyT* d_out_keys,
  const IndexT* d_out_indices,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t k,
  const cuda::std::int64_t* d_seg_sizes);

namespace detail
{
// Begin offset of segment i in a buffer whose segments start every `stride` elements: begin[i] = i * stride. Also used
// (with stride == k) to build the tightly packed output-segment offsets.
struct topk_ref_segment_begin_op
{
  cuda::std::int64_t stride;

  __host__ __device__ cuda::std::int64_t operator()(cuda::std::int64_t segment_id) const
  {
    return segment_id * stride;
  }
};

// Exclusive end offset of segment i in a buffer whose segments start every `stride` elements:
// end[i] = i * stride + seg_sizes[i].
struct topk_ref_segment_end_op
{
  cuda::std::int64_t stride;
  const cuda::std::int64_t* seg_sizes;

  __host__ __device__ cuda::std::int64_t operator()(cuda::std::int64_t segment_id) const
  {
    return segment_id * stride + seg_sizes[segment_id];
  }
};

// Consistency + bounds predicate for arg-top-k output at flat index m = seg * k + j:
// the reported (segment-local) index must be in range and the input key there must equal the reported output key.
template <typename KeyT, typename IndexT>
struct topk_ref_index_consistent_op
{
  const KeyT* in_keys;
  const KeyT* out_keys;
  const IndexT* out_indices;
  cuda::std::int64_t in_stride;
  cuda::std::int64_t k;
  const cuda::std::int64_t* seg_sizes;

  __host__ __device__ bool operator()(cuda::std::int64_t m) const
  {
    const auto seg = m / k;
    const auto idx = static_cast<cuda::std::int64_t>(out_indices[m]);
    if (idx < 0 || idx >= seg_sizes[seg])
    {
      return false;
    }
    return in_keys[seg * in_stride + idx] == out_keys[m];
  }
};

// Flags duplicate indices: position m and m+1 in the same (stride-k) segment hold equal values.
template <typename IndexT>
struct topk_ref_adjacent_duplicate_op
{
  const IndexT* sorted;
  cuda::std::int64_t k;

  __host__ __device__ bool operator()(cuda::std::int64_t m) const
  {
    return (m % k != k - 1) && sorted[m] == sorted[m + 1];
  }
};

// Gathers the j-th item of segment `seg` from a per-segment-sorted buffer (segments every `in_stride` elements),
// addressed by a flat index m = seg * k + j over the tightly packed top-k output.
template <typename KeyT>
struct topk_ref_gather_op
{
  const KeyT* sorted_in;
  cuda::std::int64_t in_stride;
  cuda::std::int64_t k;

  __host__ __device__ KeyT operator()(cuda::std::int64_t m) const
  {
    return sorted_in[(m / k) * in_stride + (m % k)];
  }
};
} // namespace detail

template <typename KeyT>
void segmented_sort_keys(
  KeyT* d_keys,
  cuda::std::int64_t num_items,
  cuda::std::int64_t num_segments,
  const cuda::std::int64_t* d_begin_offsets,
  const cuda::std::int64_t* d_end_offsets,
  cub::detail::topk::select direction)
{
  thrust::device_vector<KeyT> d_keys_alt(num_items, thrust::no_init);
  cub::DoubleBuffer<KeyT> keys(d_keys, thrust::raw_pointer_cast(d_keys_alt.data()));

  auto run = [&](void* d_temp, size_t& temp_bytes) {
    return (direction == cub::detail::topk::select::min)
           ? cub::DeviceSegmentedRadixSort::SortKeys(
               d_temp, temp_bytes, keys, num_items, num_segments, d_begin_offsets, d_end_offsets)
           : cub::DeviceSegmentedRadixSort::SortKeysDescending(
               d_temp, temp_bytes, keys, num_items, num_segments, d_begin_offsets, d_end_offsets);
  };

  size_t temp_storage_bytes = 0;
  run(nullptr, temp_storage_bytes);
  thrust::device_vector<cuda::std::uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);
  run(thrust::raw_pointer_cast(d_temp_storage.data()), temp_storage_bytes);

  // Make sure the sorted result ends up in the original buffer. All three arguments are raw device pointers, so pin
  // the device execution policy explicitly -- otherwise thrust treats raw pointers as host iterators and memcpys
  // device memory on the host (a segfault that only surfaces when the radix passes leave the result in the alt buffer).
  if (keys.Current() != d_keys)
  {
    thrust::copy(thrust::device, keys.Current(), keys.Current() + num_items, d_keys);
  }
}

template <typename KeyT>
bool verify_segmented_topk_keys(
  const KeyT* d_in_keys,
  cuda::std::int64_t in_stride,
  const KeyT* d_out_keys,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t k,
  const cuda::std::int64_t* d_seg_sizes,
  cub::detail::topk::select direction)
{
  const cuda::std::int64_t num_in_items = num_segments * in_stride;

  // Begin offsets are affine (i * in_stride); end offsets add the live segment size. Materialize both so the sort sees
  // a single concrete `const int64_t*` offset type.
  thrust::device_vector<cuda::std::int64_t> d_in_begin(num_segments + 1);
  thrust::device_vector<cuda::std::int64_t> d_in_end(num_segments);
  thrust::tabulate(d_in_begin.begin(), d_in_begin.end(), detail::topk_ref_segment_begin_op{in_stride});
  thrust::tabulate(d_in_end.begin(), d_in_end.end(), detail::topk_ref_segment_end_op{in_stride, d_seg_sizes});

  // Reference: sort each input segment's live region.
  thrust::device_vector<KeyT> reference(d_in_keys, d_in_keys + num_in_items);
  segmented_sort_keys(
    thrust::raw_pointer_cast(reference.data()),
    num_in_items,
    num_segments,
    thrust::raw_pointer_cast(d_in_begin.data()),
    thrust::raw_pointer_cast(d_in_end.data()),
    direction);

  // The top-k output is unordered, so sort each (tightly packed, stride k) output segment the same way before
  // comparing. Begin offsets are i * k; end offsets are the begin offsets shifted by one.
  thrust::device_vector<KeyT> sorted_out(d_out_keys, d_out_keys + num_segments * k);
  thrust::device_vector<cuda::std::int64_t> d_out_offsets(num_segments + 1);
  thrust::tabulate(d_out_offsets.begin(), d_out_offsets.end(), detail::topk_ref_segment_begin_op{k});
  segmented_sort_keys(
    thrust::raw_pointer_cast(sorted_out.data()),
    num_segments * k,
    num_segments,
    thrust::raw_pointer_cast(d_out_offsets.data()),
    thrust::raw_pointer_cast(d_out_offsets.data()) + 1,
    direction);

  // The expected top-k are the leading k items of each sorted reference segment, gathered on the fly.
  auto reference_topk = cuda::make_transform_iterator(
    cuda::make_counting_iterator(cuda::std::int64_t{0}),
    detail::topk_ref_gather_op<KeyT>{thrust::raw_pointer_cast(reference.data()), in_stride, k});

  return thrust::equal(sorted_out.cbegin(), sorted_out.cend(), reference_topk);
}

template <typename KeyT, typename IndexT>
bool verify_segmented_topk_indices(
  const KeyT* d_in_keys,
  cuda::std::int64_t in_stride,
  const KeyT* d_out_keys,
  const IndexT* d_out_indices,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t k,
  const cuda::std::int64_t* d_seg_sizes)
{
  // Consistency + bounds: in_keys[seg * stride + idx] == out_keys[m], with 0 <= idx < seg_size.
  const bool consistent = thrust::all_of(
    cuda::make_counting_iterator(cuda::std::int64_t{0}),
    cuda::make_counting_iterator(num_segments * k),
    detail::topk_ref_index_consistent_op<KeyT, IndexT>{d_in_keys, d_out_keys, d_out_indices, in_stride, k, d_seg_sizes});
  if (!consistent)
  {
    return false;
  }

  // Uniqueness: no repeated index within a segment. Sort each output segment, then flag equal neighbors.
  thrust::device_vector<IndexT> sorted_idx(d_out_indices, d_out_indices + num_segments * k);
  thrust::device_vector<cuda::std::int64_t> d_out_offsets(num_segments + 1);
  thrust::tabulate(d_out_offsets.begin(), d_out_offsets.end(), detail::topk_ref_segment_begin_op{k});
  segmented_sort_keys(
    thrust::raw_pointer_cast(sorted_idx.data()),
    num_segments * k,
    num_segments,
    thrust::raw_pointer_cast(d_out_offsets.data()),
    thrust::raw_pointer_cast(d_out_offsets.data()) + 1,
    cub::detail::topk::select::min);

  const IndexT* sorted_ptr   = thrust::raw_pointer_cast(sorted_idx.data());
  const auto duplicate_count = thrust::count_if(
    cuda::make_counting_iterator(cuda::std::int64_t{0}),
    cuda::make_counting_iterator(num_segments * k - 1),
    detail::topk_ref_adjacent_duplicate_op<IndexT>{sorted_ptr, k});

  return duplicate_count == 0;
}

// Suppress implicit instantiation in consuming translation units for the closed native type set; these are compiled
// once into `cccl.c2h.core`. Keep this list in sync with the explicit instantiations in device_topk_reference.cu.
#define C2H_TOPK_REF_EXTERN_KEY(KeyT)                                                              \
  extern template void segmented_sort_keys<KeyT>(                                                  \
    KeyT*,                                                                                         \
    cuda::std::int64_t,                                                                            \
    cuda::std::int64_t,                                                                            \
    const cuda::std::int64_t*,                                                                     \
    const cuda::std::int64_t*,                                                                     \
    cub::detail::topk::select);                                                                    \
  extern template bool verify_segmented_topk_keys<KeyT>(                                            \
    const KeyT*,                                                                                    \
    cuda::std::int64_t,                                                                             \
    const KeyT*,                                                                                    \
    cuda::std::int64_t,                                                                             \
    cuda::std::int64_t,                                                                             \
    const cuda::std::int64_t*,                                                                      \
    cub::detail::topk::select)

#define C2H_TOPK_REF_EXTERN_INDICES(KeyT, IndexT)                                                  \
  extern template bool verify_segmented_topk_indices<KeyT, IndexT>(                                 \
    const KeyT*,                                                                                    \
    cuda::std::int64_t,                                                                             \
    const KeyT*,                                                                                    \
    const IndexT*,                                                                                  \
    cuda::std::int64_t,                                                                             \
    cuda::std::int64_t,                                                                             \
    const cuda::std::int64_t*)

C2H_TOPK_REF_EXTERN_KEY(cuda::std::int8_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::int16_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::int32_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::int64_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::uint8_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::uint16_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::uint32_t);
C2H_TOPK_REF_EXTERN_KEY(cuda::std::uint64_t);
C2H_TOPK_REF_EXTERN_KEY(float);
C2H_TOPK_REF_EXTERN_KEY(double);

C2H_TOPK_REF_EXTERN_INDICES(float, cuda::std::int32_t);
C2H_TOPK_REF_EXTERN_INDICES(float, cuda::std::int64_t);

#undef C2H_TOPK_REF_EXTERN_KEY
#undef C2H_TOPK_REF_EXTERN_INDICES

} // namespace c2h
