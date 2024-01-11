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

using namespace cub;

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

  int num_items {};
  thrust::device_vector<MaskedValueT> d_values;
  thrust::host_vector<KeyT> h_keys;
  thrust::host_vector<MaskedValueT> h_values;

public:
  Input(const thrust::host_vector<int> &h_segment_sizes)
      : d_segment_sizes(h_segment_sizes)
      , d_offsets(d_segment_sizes.size() + 1)
      , h_offsets(d_segment_sizes.size() + 1)
      , num_items(static_cast<int>(
          thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end())))
      , d_values(num_items)
      , h_keys(num_items)
      , h_values(num_items)
  {
    update();
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

  thrust::device_vector<MaskedValueT> &get_d_values_vec()
  {
    return d_values;
  }

  const thrust::host_vector<int>& get_h_offsets()
  {
    return h_offsets;
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

    ReversedIota<KeyT> generator{keys_output, offsets};

    for (int i = 0; i < get_num_segments(); i++)
    {
      generator(i);
    }
  }
};

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


template <typename KeyT,
          typename ValueT>
void HostReferenceSort(unsigned int num_segments,
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

    if (segment_begin > segment_end)
    {
      std::cout << "segment_begin > segment_end" << std::endl;
    }

    if (segment_begin < 0)
    {
      std::cout << "segment_begin < 0" << std::endl;
    }

    if (segment_end > static_cast<int>(h_keys.size()))
    {
      std::cout << "segment_end > h_keys.size()" << std::endl;
    }

    if (segment_begin > static_cast<int>(h_keys.size()))
    {
      std::cout << "segment_begin > h_keys.size()" << std::endl;
    }

    thrust::stable_sort(h_keys.begin() + segment_begin,
                        h_keys.begin() + segment_end,
                        thrust::greater<KeyT>{});
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

  RandomizeInput(h_keys, h_values);
  HostReferenceSort(input.get_num_segments(), h_offsets, h_keys, h_values);
}

template <typename KeyT,
          typename ValueT>
Input<KeyT, ValueT> GenRandomInput(int max_items,
                                   int min_segments,
                                   int max_segments)
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

  return Input<KeyT, ValueT>{segment_sizes};
}

template <typename KeyT,
          typename ValueT>
void RandomTest(int min_segments,
                int max_segments)
{
  constexpr int max_items = 10000000;

  for (int iteration = 0; iteration < 2; iteration++)
  {
    Input<KeyT, ValueT> edge_cases = GenRandomInput<KeyT, ValueT>(max_items,
                                                                  min_segments,
                                                                  max_segments);

    InputTestRandom(edge_cases);
  }
}

int main(int argc, char** argv)
{
  // /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DCUB_DETAIL_DEBUG_ENABLE_SYNC -DTEST_LAUNCH=0 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DVAR_IDX=0 -D_CCCL_NO_SYSTEM_HEADER -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/cub/test -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/cub/cub/cmake/../.. -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/libcudacxx/lib/cmake/libcudacxx/../../../include -I/home/gevtushenko/${ccclroot}/senior-zero/cccl/thrust/thrust/cmake/../.. -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -O3 -DNDEBUG /home/gevtushenko/src/senior-zero/cccl/cub/test/test_device_segmented_sort.cu -o a.out -ccbin=g++-12
  RandomTest<bool, std::uint64_t>(1 << 9, 1 << 19);

  return 0;
}
