/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/std/optional>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and runing it twice.
#if TEST_LAUNCH == 0
struct block_size_check_t
{
  int* ptr;

  __device__ int operator()(int a, int b)
  {
    *ptr = blockDim.x;
    return a + b;
  }
};

struct block_size_retreiver_t
{
  int* ptr;

  template <class ActivePolicyT>
  cudaError_t Invoke()
  {
    *ptr = ActivePolicyT::SingleTilePolicy::BLOCK_THREADS;
    return cudaSuccess;
  }
};

TEST_CASE("Device reduce works with default environment", "[reduce][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;
  using policy_t    = cub::detail::reduce::default_tuning::type<value_t, offset_t, block_size_check_t>::MaxPolicy;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  int ptx_version{};
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version, current_device));

  int target_block_size{};
  block_size_retreiver_t block_size_retreiver{&target_block_size};
  REQUIRE(cudaSuccess == policy_t::Invoke(ptx_version, block_size_retreiver));

  num_items_t num_items = 1;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = thrust::make_constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.begin(), num_items, block_size_check, value_t{0}));
  REQUIRE(d_out[0] == num_items);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

template <int BlockThreads>
struct reduce_tuning : cub::detail::reduce::reduce_tuning<reduce_tuning<BlockThreads>>
{
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct type
  {
    struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>
    {
      struct ReducePolicy
      {
        static constexpr int VECTOR_LOAD_LENGTH = 1;

        static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BLOCK_REDUCE_WARP_REDUCTIONS;

        static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;

        static constexpr int ITEMS_PER_THREAD = 1;
        static constexpr int BLOCK_THREADS    = BlockThreads;
      };

      using SingleTilePolicy      = ReducePolicy;
      using SegmentedReducePolicy = ReducePolicy;
    };

    using MaxPolicy = Policy500;
  };
};

struct get_scan_tuning_query_t
{};

struct scan_tuning
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(const get_scan_tuning_query_t&) const noexcept
  {
    return *this;
  }

  // Make sure this is not used
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct type
  {};
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device reduce can be tuned", "[reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 1;
  auto d_in      = thrust::make_constant_iterator(1);
  auto d_out     = thrust::device_vector<int>(1);

  // We are expecting that `scan_tuning` is ignored
  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{}, scan_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.begin(), num_items, block_size_check, 0, env));
  REQUIRE(d_out[0] == num_items);
  REQUIRE(d_block_size[0] == target_block_size);
}
#endif

using requirements =
  c2h::type_list<cuda::execution::determinism::run_to_run_t, cuda::execution::determinism::not_guaranteed_t>;

C2H_TEST("Device reduce uses environment", "[reduce][device]", requirements)
{
  using determinism_t = c2h::get<0, TestType>;
  using accumulator_t = int;
  using op_t          = cuda::std::plus<>;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = ::cuda::std::__identity;
  using policy_t      = cub::detail::reduce::policy_hub<accumulator_t, offset_t, op_t>::MaxPolicy;
  using init_t        = accumulator_t;

  num_items_t num_items = GENERATE(1 << 4, 1 << 24);
  auto d_in             = thrust::make_constant_iterator(1);
  auto d_out            = thrust::device_vector<accumulator_t>(1);

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    // TODO(gevtushenko): split `not_guaranteed` kernels once atomic reduce is merged
    if constexpr (std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>
                  || std::is_same_v<determinism_t, cuda::execution::determinism::not_guaranteed_t>)
    {
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            decltype(d_in),
            decltype(d_out.begin()),
            offset_t,
            op_t,
            init_t,
            accumulator_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceKernel<policy_t, decltype(d_in), offset_t, op_t, accumulator_t, transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            accumulator_t*,
            decltype(d_out.begin()),
            int, // always used with int offset
            op_t,
            init_t,
            accumulator_t>)};
    }
    else
    {
      // TODO(gevtushenko): add `gpu_to_gpu` kernels once RFA is merged
      FAIL("Only run_to_run and not_guaranteed determinism are supported at the moment");
      return cuda::std::array<void*, 0>{};
    }
  }();

  init_t init = 0;

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Reduce(
            nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, cuda::std::plus<>{}, init));

  // Equivalent to `cuexec::require(cuexec::determinism::run_to_run)` and
  //               `cuexec::require(cuexec::determinism::not_guaranteed)`
  auto env = stdexec::env{cuda::execution::require(determinism_t{}), // determinism
                          allowed_kernels(kernels), // allowed kernels for the given determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  // TODO(gevtushenko): how to check if given requirement is met?
  device_reduce(d_in, d_out.begin(), num_items, cuda::std::plus<>{}, init, env);

  REQUIRE(d_out[0] == num_items);
}
