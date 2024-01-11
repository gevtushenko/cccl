/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_segmented_sort.cuh>

#include <nv/target>

#include <test_util.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <fstream>

#define TEST_HALF_T !_NVHPC_CUDA

#define TEST_BF_T !_NVHPC_CUDA

#if TEST_HALF_T
#include <cuda_fp16.h>
#endif

#if TEST_BF_T
#include <cuda_bf16.h>
#endif

using namespace cub;

template <typename T>
struct UnwrapHalfAndBfloat16
{
  using Type = T;
};

#if TEST_HALF_T
template <>
struct UnwrapHalfAndBfloat16<half_t>
{
  using Type = __half;
};
#endif

#if TEST_BF_T
template <>
struct UnwrapHalfAndBfloat16<bfloat16_t>
{
  using Type = __nv_bfloat16;
};
#endif

static constexpr int MAX_ITERATIONS = 2;


class SizeGroupDescription
{
public:
  SizeGroupDescription(const int segments,
                       const int segment_size)
      : segments(segments)
      , segment_size(segment_size)
  {}

  int segments {};
  int segment_size {};
};

template <typename KeyT>
struct SegmentChecker
{
  const KeyT *sorted_keys {};
  const int *offsets {};

  SegmentChecker(const KeyT *sorted_keys,
                 const int *offsets)
    : sorted_keys(sorted_keys)
    , offsets(offsets)
  {}

  bool operator()(int segment_id)
  {
    const int segment_begin = offsets[segment_id];
    const int segment_end = offsets[segment_id + 1];

    int counter = 0;
    for (int i = segment_begin; i < segment_end; i++)
    {
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT>
struct DescendingSegmentChecker
{
  const KeyT *sorted_keys{};
  const int *offsets{};

  DescendingSegmentChecker(const KeyT *sorted_keys,
                           const int *offsets)
      : sorted_keys(sorted_keys)
      , offsets(offsets)
  {}

  bool operator()(int segment_id)
  {
    const int segment_begin = offsets[segment_id];
    const int segment_end   = offsets[segment_id + 1];

    int counter = 0;
    for (int i = segment_end - 1; i >= segment_begin; i--)
    {
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT>
struct ReversedIota
{
  KeyT *data {};
  const int *offsets {};

  ReversedIota(KeyT *data,
               const int *offsets)
    : data(data)
    , offsets(offsets)
  {}

  void operator()(int segment_id) const
  {
    const int segment_begin = offsets[segment_id];
    const int segment_end = offsets[segment_id + 1];
    const int segment_size = segment_end - segment_begin;

    int count = 0;
    for (int i = segment_begin; i < segment_end; i++)
    {
      data[i] = static_cast<KeyT>(segment_size - 1 - count++);
    }
  }
};


template <typename KeyT>
struct Iota
{
  KeyT *data{};
  const int *offsets{};

  Iota(KeyT *data, const int *offsets)
      : data(data)
      , offsets(offsets)
  {}

  void operator()(int segment_id) const
  {
    const int segment_begin = offsets[segment_id];
    const int segment_end   = offsets[segment_id + 1];

    int count = 0;
    for (int i = segment_begin; i < segment_end; i++)
    {
      data[i] = static_cast<KeyT>(count++);
    }
  }
};


template <typename KeyT,
          typename ValueT = cub::NullType>
class Input
{
  thrust::default_random_engine random_engine;
  thrust::device_vector<int> d_segment_sizes;
  thrust::device_vector<int> d_offsets;
  thrust::host_vector<int> h_offsets;

  using MaskedValueT = cub::detail::conditional_t<
    std::is_same<ValueT, cub::NullType>::value, KeyT, ValueT>;

  bool reverse {};
  int num_items {};
  thrust::device_vector<KeyT> d_keys;
  thrust::device_vector<MaskedValueT> d_values;
  thrust::host_vector<KeyT> h_keys;
  thrust::host_vector<MaskedValueT> h_values;

public:
  Input(bool reverse, const thrust::host_vector<int> &h_segment_sizes)
      : d_segment_sizes(h_segment_sizes)
      , d_offsets(d_segment_sizes.size() + 1)
      , h_offsets(d_segment_sizes.size() + 1)
      , reverse(reverse)
      , num_items(static_cast<int>(
          thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end())))
      , d_keys(num_items)
      , d_values(num_items)
      , h_keys(num_items)
      , h_values(num_items)
  {
    update();
  }

  Input(thrust::host_vector<int> &h_offsets)
    : d_offsets(h_offsets)
    , h_offsets(h_offsets)
    , reverse(false)
    , num_items(h_offsets.back())
    , d_keys(num_items)
    , d_values(num_items)
  {
  }

  void shuffle()
  {
    thrust::shuffle(d_segment_sizes.begin(), d_segment_sizes.end(), random_engine);

    update();
  }

  int get_num_items() const
  {
    return num_items;
  }

  int get_num_segments() const
  {
    return static_cast<unsigned int>(d_segment_sizes.size());
  }

  const KeyT *get_d_keys() const
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  thrust::device_vector<KeyT> &get_d_keys_vec()
  {
    return d_keys;
  }

  thrust::device_vector<MaskedValueT> &get_d_values_vec()
  {
    return d_values;
  }

  KeyT *get_d_keys()
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  const thrust::host_vector<int>& get_h_offsets()
  {
    return h_offsets;
  }

  MaskedValueT *get_d_values()
  {
    return thrust::raw_pointer_cast(d_values.data());
  }

  const int *get_d_offsets() const
  {
    return thrust::raw_pointer_cast(d_offsets.data());
  }

  template <typename T>
  bool check_output_implementation(const T *keys_output)
  {
    const int *offsets = thrust::raw_pointer_cast(h_offsets.data());

    if (reverse)
    {
      DescendingSegmentChecker<T> checker{keys_output, offsets};

      for (int i = 0; i < get_num_segments(); i++)
      {
        if (!checker(i))
        {
          return false;
        }
      }
    }
    else
    {
      SegmentChecker<T> checker{keys_output, offsets};

      for (int i = 0; i < get_num_segments(); i++)
      {
        if (!checker(i))
        {
          return false;
        }
      }
    }

    return true;
  }

  bool check_output(const KeyT *d_keys_output,
                    const MaskedValueT *d_values_output = nullptr)
  {
    KeyT *keys_output = thrust::raw_pointer_cast(h_keys.data());
    MaskedValueT *values_output = thrust::raw_pointer_cast(h_values.data());

    cudaMemcpy(keys_output,
               d_keys_output,
               sizeof(KeyT) * num_items,
               cudaMemcpyDeviceToHost);

    const bool keys_ok = check_output_implementation(keys_output);

    if (std::is_same<ValueT, cub::NullType>::value || d_values_output == nullptr)
    {
      return keys_ok;
    }

    cudaMemcpy(values_output,
               d_values_output,
               sizeof(ValueT) * num_items,
               cudaMemcpyDeviceToHost);

    const bool values_ok = check_output_implementation(values_output);

    return keys_ok && values_ok;
  }

private:
  void update()
  {
    fill_offsets();
    gen_keys();
  }

  void fill_offsets()
  {
    thrust::copy(d_segment_sizes.begin(), d_segment_sizes.end(), d_offsets.begin());
    thrust::exclusive_scan(d_offsets.begin(), d_offsets.end(), d_offsets.begin(), 0u);
    thrust::copy(d_offsets.begin(), d_offsets.end(), h_offsets.begin());
  }

  void gen_keys()
  {
    KeyT *keys_output = thrust::raw_pointer_cast(h_keys.data());
    const int *offsets = thrust::raw_pointer_cast(h_offsets.data());

    if (reverse)
    {
      Iota<KeyT> generator{keys_output, offsets};

      for (int i = 0; i < get_num_segments(); i++)
      {
        generator(i);
      }
    }
    else
    {
      ReversedIota<KeyT> generator{keys_output, offsets};

      for (int i = 0; i < get_num_segments(); i++)
      {
        generator(i);
      }
    }

    d_keys = h_keys;
    d_values = d_keys;
  }
};

template <typename KeyT,
          bool IsIntegralType = std::is_integral<KeyT>::value>
class InputDescription
{
  thrust::host_vector<int> segment_sizes;

public:
  InputDescription& add(const SizeGroupDescription &group)
  {
    if (static_cast<std::size_t>(group.segment_size) <
        static_cast<std::size_t>((std::numeric_limits<KeyT>::max)()))
    {
      for (int i = 0; i < group.segments; i++)
      {
        segment_sizes.push_back(group.segment_size);
      }
    }

    return *this;
  }

  template <typename ValueT = cub::NullType>
  Input<KeyT, ValueT> gen(bool reverse)
  {
    return Input<KeyT, ValueT>(reverse, segment_sizes);
  }
};

template <typename KeyT>
class InputDescription<KeyT, false>
{
  thrust::host_vector<int> segment_sizes;

public:
  InputDescription& add(const SizeGroupDescription &group)
  {
    for (int i = 0; i < group.segments; i++)
    {
      segment_sizes.push_back(group.segment_size);
    }

    return *this;
  }

  template <typename ValueT = cub::NullType>
  Input<KeyT, ValueT> gen(bool reverse)
  {
    return Input<KeyT, ValueT>(reverse, segment_sizes);
  }
};

constexpr bool keys_only = false;
constexpr bool pairs = true;

constexpr bool ascending = false;
constexpr bool descending = true;

constexpr bool pointers = false;
constexpr bool double_buffer = true;

constexpr bool unstable = false;
constexpr bool stable = true;

template <typename KeyT,
          typename ValueT>
void TestSameSizeSegments(int segment_size,
                          int segments,
                          bool skip_values = false)
{
  const int num_items = segment_size * segments;

  thrust::device_vector<int> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   int{},
                   segment_size);

  const int *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT target_key {1};
  const ValueT target_value {42};

  thrust::device_vector<KeyT> keys_input(num_items);
  thrust::device_vector<KeyT> keys_output(num_items);

  KeyT *d_keys_input  = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(num_items);
  thrust::device_vector<ValueT> values_output(num_items);

  thrust::host_vector<KeyT> host_keys(num_items);
  thrust::host_vector<ValueT> host_values(num_items);

  ValueT *d_values_input  = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs: { keys_only, pairs })
    {
      if (sort_pairs)
      {
        if (skip_values)
        {
          continue;
        }
      }

      for (bool sort_descending: { ascending, descending })
      {
        for (bool sort_buffers: { pointers, double_buffer })
        {
          cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
          cub::DoubleBuffer<ValueT> values_buffer(nullptr, nullptr);
          values_buffer.selector = 1;

          thrust::fill(keys_input.begin(), keys_input.end(), target_key);
          thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

          if (sort_pairs)
          {
            if (sort_buffers)
            {
              thrust::fill(values_input.begin(), values_input.end(), ValueT{});
              thrust::fill(values_output.begin(), values_output.end(), target_value);
            }
            else
            {
              thrust::fill(values_input.begin(), values_input.end(), target_value);
              thrust::fill(values_output.begin(), values_output.end(), ValueT{});
            }
          }

          // If temporary storage size is defined by extra keys storage
          {
            host_keys = keys_buffer.selector || !sort_buffers ? keys_output
                                                              : keys_input;
            const std::size_t items_selected =
              thrust::count(host_keys.begin(), host_keys.end(), target_key);
          }

          if (sort_pairs)
          {
            host_values = values_buffer.selector || !sort_buffers
                            ? values_output
                            : values_input;
            const std::size_t items_selected =
              thrust::count(host_values.begin(),
                            host_values.end(),
                            target_value);
          }
        }
      }
    }
  }
}


template <typename KeyT,
          typename ValueT>
void InputTest(bool sort_descending,
               Input<KeyT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  for (bool stable_sort: { unstable, stable })
  {
    for (bool sort_pairs : { keys_only, pairs })
    {
      for (bool sort_buffers : {pointers, double_buffer})
      {
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
        {
          thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
          thrust::fill(values_output.begin(), values_output.end(), ValueT{});

          input.shuffle();
        }
      }
    }
  }
}

struct ComparisonPredicate
{
  template <typename T>
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs == rhs;
  }

  __host__ __device__ bool operator()(const half_t &lhs, const half_t &rhs) const
  {
    return lhs.raw() == rhs.raw();
  }
};

template <typename T>
bool compare_two_outputs(const thrust::host_vector<int> &offsets,
                         const thrust::host_vector<T> &lhs,
                         const thrust::host_vector<T> &rhs)
{
  const auto num_segments = static_cast<unsigned int>(offsets.size() - 1);

  for (std::size_t segment_id = 0; segment_id < num_segments; segment_id++)
  {
    auto lhs_begin = lhs.cbegin() + offsets[segment_id];
    auto lhs_end = lhs.cbegin() + offsets[segment_id + 1];
    auto rhs_begin = rhs.cbegin() + offsets[segment_id];

    auto err = thrust::mismatch(lhs_begin, lhs_end, rhs_begin, ComparisonPredicate{});

    if (err.first != lhs_end)
    {
      const auto idx = thrust::distance(lhs_begin, err.first);
      const auto segment_size = std::distance(lhs_begin, lhs_end);

      std::cerr << "Mismatch in segment " << segment_id
                << " at position " << idx << " / " << segment_size
                << ": "
                << static_cast<std::uint64_t>(lhs_begin[idx]) << " vs "
                << static_cast<std::uint64_t>(rhs_begin[idx]) << " ("
                << typeid(lhs_begin[idx]).name() << ")" << std::endl;

      return false;
    }
  }

  return true;
}

template <typename ValueT>
void RandomizeInput(thrust::host_vector<bool> &h_keys,
                    thrust::host_vector<ValueT> &h_values)
{
  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    h_keys[i] = RandomValue((std::numeric_limits<std::uint8_t>::max)()) > 128;
    h_values[i] = RandomValue((std::numeric_limits<ValueT>::max)());
  }
}

template <typename KeyT,
          typename ValueT>
void RandomizeInput(thrust::host_vector<KeyT> &h_keys,
                    thrust::host_vector<ValueT> &h_values)
{
  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    h_keys[i] = RandomValue((std::numeric_limits<KeyT>::max)());
    h_values[i] = RandomValue((std::numeric_limits<ValueT>::max)());
  }
}

#if TEST_HALF_T
void RandomizeInput(thrust::host_vector<half_t> &h_keys,
                    thrust::host_vector<std::uint32_t> &h_values)
{
  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    h_keys[i] = RandomValue((std::numeric_limits<int>::max)());
    h_values[i] = RandomValue((std::numeric_limits<std::uint32_t>::max)());
  }
}
#endif

#if TEST_BF_T
void RandomizeInput(thrust::host_vector<bfloat16_t> &h_keys,
                    thrust::host_vector<std::uint32_t> &h_values)
{
  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    h_keys[i] = RandomValue((std::numeric_limits<int>::max)());
    h_values[i] = RandomValue((std::numeric_limits<std::uint32_t>::max)());
  }
}
#endif



template <typename KeyT,
          typename ValueT>
void HostReferenceSort(bool sort_pairs,
                       bool sort_descending,
                       unsigned int num_segments,
                       const thrust::host_vector<int> &h_offsets,
                       thrust::host_vector<KeyT> &h_keys,
                       thrust::host_vector<ValueT> &h_values)
{
  for (unsigned int segment_i = 0;
       segment_i < num_segments;
       segment_i++)
  {
    const int segment_begin = h_offsets[segment_i];
    const int segment_end   = h_offsets[segment_i + 1];

    if (sort_pairs)
    {
      if (sort_descending)
      {
        thrust::stable_sort_by_key(h_keys.begin() + segment_begin,
                                   h_keys.begin() + segment_end,
                                   h_values.begin() + segment_begin,
                                   thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort_by_key(h_keys.begin() + segment_begin,
                                   h_keys.begin() + segment_end,
                                   h_values.begin() + segment_begin);
      }
    }
    else
    {
      if (sort_descending)
      {
        thrust::stable_sort(h_keys.begin() + segment_begin,
                            h_keys.begin() + segment_end,
                            thrust::greater<KeyT>{});
      }
      else
      {
        thrust::stable_sort(h_keys.begin() + segment_begin,
                            h_keys.begin() + segment_end);
      }
    }
  }
}


template <typename KeyT,
          typename ValueT>
void InputTestRandom(Input<KeyT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  const thrust::host_vector<int> &h_offsets = input.get_h_offsets();

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    RandomizeInput(h_keys, h_values);

    input.get_d_keys_vec()   = h_keys;
    input.get_d_values_vec() = h_values;

    HostReferenceSort(false, false, input.get_num_segments(), h_offsets, h_keys, h_values);

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT>
Input<KeyT, ValueT> GenRandomInput(int max_items,
                                   int min_segments,
                                   int max_segments,
                                   bool descending)
{
  int items_generated {};
  const int segments_num = RandomValue(max_segments) + min_segments;

  thrust::host_vector<int> segment_sizes;
  segment_sizes.reserve(segments_num);

  constexpr int max_segment_size = 6000;

  for (int segment_id = 0; segment_id < segments_num; segment_id++)
  {
    const int segment_size_raw = RandomValue(max_segment_size);
    const int segment_size     = segment_size_raw > 0 ? segment_size_raw : 0;

    if (segment_size + items_generated > max_items)
    {
      break;
    }

    items_generated += segment_size;
    segment_sizes.push_back(segment_size);
  }

  return Input<KeyT, ValueT>{descending, segment_sizes};
}

template <typename KeyT,
          typename ValueT>
void RandomTest(int min_segments,
                int max_segments)
{
  constexpr int max_items = 10000000;

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    Input<KeyT, ValueT> edge_cases = GenRandomInput<KeyT, ValueT>(max_items,
                                                                  min_segments,
                                                                  max_segments,
                                                                  descending);

    InputTestRandom(edge_cases);
  }
}

int main(int argc, char** argv)
{
  // /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DCUB_DETAIL_DEBUG_ENABLE_SYNC -DTEST_LAUNCH=0 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DVAR_IDX=0 -D_CCCL_NO_SYSTEM_HEADER -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/cub/test -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/cub/cub/cmake/../.. -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/libcudacxx/lib/cmake/libcudacxx/../../../include -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/thrust/thrust/cmake/../.. -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -O3 -DNDEBUG /home/gevtushenko/src/senior-zero/cccl/cub/test/test_device_segmented_sort.cu -o a.out -ccbin=g++-12
  RandomTest<bool, std::uint64_t>(1 << 9, 1 << 19);

  return 0;
}
