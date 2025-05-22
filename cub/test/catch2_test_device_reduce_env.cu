/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

// #include <cuda/experimental/__execution/env.cuh>
// #include <cuda/experimental/memory_resource.cuh>
// #include <cuda/experimental/stream.cuh>

#include <cuda/std/optional>

#include <cstdint>

#include <c2h/catch2_test_helper.h>

struct stream_registry_factory_t
{
  cuda::std::optional<cudaStream_t> m_stream;

  thrust::cuda_cub::detail::triple_chevron
  operator()(dim3 grid, dim3 block, size_t shared_mem, cudaStream_t stream, bool dependent_launch = false) const
  {
    if (m_stream)
    {
      REQUIRE(stream == m_stream);
    }
    return thrust::cuda_cub::detail::triple_chevron(grid, block, shared_mem, stream, dependent_launch);
  }

  cudaError_t PtxVersion(int& version)
  {
    return cub::PtxVersion(version);
  }

  cudaError_t MultiProcessorCount(int& sm_count) const
  {
    int device_ordinal;
    cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
  }

  template <typename Kernel>
  cudaError_t MaxSmOccupancy(int& sm_occupancy, Kernel kernel_ptr, int block_size, int dynamic_smem_bytes = 0)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_ptr, block_size, dynamic_smem_bytes);
  }

  cudaError_t MaxGridDimX(int& max_grid_dim_x) const
  {
    int device_ordinal;
    cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get max grid dimension
    return cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device_ordinal);
  }
};

// singleton
stream_registry_factory_t& get_stream_registry_factory()
{
  static stream_registry_factory_t factory;
  return factory;
}

struct stream_scope
{
  stream_scope(cudaStream_t stream)
  {
    get_stream_registry_factory().m_stream = stream;
  }

  ~stream_scope()
  {
    get_stream_registry_factory().m_stream = cuda::std::nullopt;
  }
};

TEST_CASE("Device reduce works with default environment", "[reduce][device]")
{
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5};
  thrust::device_vector<int> d_out(1);

  cudaError_t err = cub::DeviceReduce::Reduce(d_in.begin(), d_out.begin(), d_in.size(), cuda::std::plus<>{}, 0);
  REQUIRE(err == cudaSuccess);

  REQUIRE(d_out[0] == 15);
}

TEST_CASE("Device reduce works with cudax environment", "[reduce][device]")
{
  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  thrust::device_vector<int> d_in{1, 2, 3, 4, 5};
  thrust::device_vector<int> d_out(1);

  cuda::std::execution::prop env{cuda::get_stream, stream};

  {
    stream_scope scope(stream);
    REQUIRE(
      cudaSuccess == cub::DeviceReduce::Reduce(d_in.begin(), d_out.begin(), d_in.size(), cuda::std::plus<>{}, 0, env));
  }

  REQUIRE(d_out[0] == 15);

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}
