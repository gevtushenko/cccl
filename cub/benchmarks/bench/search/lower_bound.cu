// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/binary_search_helpers.cuh>
#include <cub/device/device_find.cuh>
#include <cub/device/device_for.cuh>

#include <thrust/sort.h>

#include <cuda/std/functional>
#include <cuda/std/tuple>

#include <nvbench_helper.cuh>

// Set to 1 for CUB binary search, 0 for linear scan via ForEachN
#define USE_CUB 0

// Linear-scan lower bound mode: for each needle, walk all haystack elements and count how many are
// strictly less than the needle. This is the GPU analogue of the Triton pattern:
//   idx = sum(haystack[i] < needle for all i)
// Plugs into the same comp_wrapper_t / ForEachN machinery as cub::detail::find::lower_bound.
struct linear_lower_bound
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static ::cuda::std::ptrdiff_t
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    int idx = 0;
    for (auto it = first; it != last; ++it)
    {
      idx += comp(*it, value);
    }
    return idx;
  }
};

template <typename T>
static void verify_linear_scan(
  nvbench::state& state, const T* d_range, int elements, const T* d_values, std::size_t needles)
{
  thrust::device_vector<T> cub_result(needles);
  thrust::device_vector<T> linear_result(needles);

  size_t cub_temp_size{};
  cub::DeviceFind::LowerBound(
    nullptr, cub_temp_size, d_range, elements, d_values, needles,
    thrust::raw_pointer_cast(cub_result.data()), cuda::std::less<>{});
  thrust::device_vector<uint8_t> cub_temp(cub_temp_size);
  cub::DeviceFind::LowerBound(
    thrust::raw_pointer_cast(cub_temp.data()), cub_temp_size, d_range, elements, d_values, needles,
    thrust::raw_pointer_cast(cub_result.data()), cuda::std::less<>{});

  auto linear_op = cub::detail::find::make_comp_wrapper<linear_lower_bound>(d_range, elements, cuda::std::less<>{});
  auto zip       = ::cuda::make_zip_iterator(d_values, thrust::raw_pointer_cast(linear_result.data()));
  size_t linear_temp_size{};
  cub::DeviceFor::ForEachN(nullptr, linear_temp_size, zip, needles, linear_op);
  thrust::device_vector<uint8_t> linear_temp(linear_temp_size);
  cub::DeviceFor::ForEachN(thrust::raw_pointer_cast(linear_temp.data()), linear_temp_size, zip, needles, linear_op);

  if (cub_result != linear_result)
  {
    state.skip("linear scan results do not match CUB lower_bound");
  }
}

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto needles  = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto elements = 8;

  thrust::device_vector<T> data = generate(elements + needles);
  thrust::device_vector<T> result(needles);
  thrust::sort(data.begin(), data.begin() + elements);

  state.add_global_memory_reads<T>(needles + elements, "DataSize");
  state.add_global_memory_writes<T>(needles);

  const T* d_range  = thrust::raw_pointer_cast(data.data());
  const T* d_values = d_range + elements;
  T* d_output       = thrust::raw_pointer_cast(result.data());

#if USE_CUB
  size_t temp_storage_size{};
  cub::DeviceFind::LowerBound(
    nullptr, temp_storage_size, d_range, elements, d_values, needles, d_output, cuda::std::less<>{});

  thrust::device_vector<uint8_t> temp_storage(temp_storage_size);
  void* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
#else
  auto op = cub::detail::find::make_comp_wrapper<linear_lower_bound>(d_range, elements, cuda::std::less<>{});

  size_t temp_storage_size{};
  cub::DeviceFor::ForEachN(nullptr, temp_storage_size, ::cuda::make_zip_iterator(d_values, d_output), needles, op);

  thrust::device_vector<uint8_t> temp_storage(temp_storage_size);
  void* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
#endif

  verify_linear_scan(state, d_range, elements, d_values, needles);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
#if USE_CUB
    cub::DeviceFind::LowerBound(
      d_temp_storage,
      temp_storage_size,
      d_range,
      elements,
      d_values,
      needles,
      d_output,
      cuda::std::less<>{},
      launch.get_stream());
#else
    cub::DeviceFor::ForEachN(
      d_temp_storage, temp_storage_size, ::cuda::make_zip_iterator(d_values, d_output), needles, op,
      launch.get_stream());
#endif
  });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
