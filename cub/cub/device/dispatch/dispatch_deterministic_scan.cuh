/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace run_to_run_deterministic_scan
{

template <class AccumT>
struct policy_hub_t 
{
  /// SM35
  struct policy_350_t : ChainedPolicy<350, policy_350_t, policy_350_t>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    using ReducePolicy = AgentReducePolicy<threads_per_block,
                                           items_per_thread,
                                           AccumT,
                                           items_per_vec_load,
                                           BLOCK_REDUCE_WARP_REDUCTIONS,
                                           LOAD_DEFAULT>;
  };

  using MaxPolicy = policy_350_t;
};

template <class ChainedPolicyT,
          class InputIteratorT,
          class ScanOpT,
          class OffsetT,
          class AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void reduce_kernel(
    InputIteratorT d_in,
    AccumT *d_tile_aggregates,
    ScanOpT scan_op,
    OffsetT num_items)
{
  // thread block reduces tile to one aggregate
  using policy_t = typename ChainedPolicyT::ActivePolicy::ReducePolicy;

  constexpr auto tile_items = policy_t::BLOCK_THREADS * policy_t::ITEMS_PER_THREAD;

  const auto tile_id    = static_cast<OffsetT>(blockIdx.x);
  const auto tile_begin = tile_id * tile_items;
  const auto tile_end   = tile_begin + tile_items ? num_items : tile_begin + tile_items;

  using AgentReduceT =
    AgentReduce<typename ChainedPolicyT::ActivePolicy::ReducePolicy,
                InputIteratorT,
                AccumT*,
                OffsetT,
                ScanOpT,
                AccumT,
                ::cuda::std::__identity>;

  __shared__ typename AgentReduceT::TempStorage temp_storage;

  AccumT block_aggregate =
    AgentReduceT(temp_storage, d_in, scan_op, ::cuda::std::__identity{}).ConsumeTiles(tile_begin, tile_end);

  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy(d_tile_aggregates + tile_id, block_aggregate);
  }
}

template <class ChainedPolicyT, class InitValueT, class ScanOpT, class AccumT, class OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void tile_scan_kernel(
    AccumT* d_tile_aggregates, ScanOpT scan_op, InitValueT init_value, OffsetT num_tiles)
{
  using policy_t                  = typename ChainedPolicyT::ActivePolicy::ReducePolicy;
  constexpr auto items_per_thread = policy_t::BLOCK_THREADS;
  constexpr auto block_threads    = policy_t::ITEMS_PER_THREAD;
  constexpr bool is_inclusive     = ::cuda::std::is_same<InitValueT, NullType>::value;

  // single thread block computes prefix sum of tile aggregates
  using real_init_value_t = typename InitValueT::value_type;
  using scan_t            = BlockScan<AccumT, block_threads>;
  using temp_storage_t    = typename scan_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

  constexpr auto tile_items = block_threads * items_per_thread;

  real_init_value_t real_init_value = init_value;
  BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);
  prefix_op.running_total = real_init_value;

  OffsetT offset = 0;
  while (offset < num_tiles) 
  {
    const auto valid_items = CUB_MIN(num_tiles - offset, tile_items);

    // TODO block load
    AccumT items[items_per_thread];
    LoadDirectBlocked(threadIdx.x, d_tile_aggregates + offset, items, valid_items);

    if (is_inclusive)
    {
      scan_t(temp_storage).InclusiveScan(items, items, scan_op, prefix_op);
    }
    else
    {
      scan_t(temp_storage).ExclusiveScan(items, items, scan_op, prefix_op);
    }
    CTA_SYNC();

    StoreDirectBlocked(threadIdx.x, d_tile_aggregates + offset, items, valid_items);
  }
}

// TODO Iterate tiles in reverse order to utilize some cache left from the first pass
template <class ChainedPolicyT,
          class InputIteratorT,
          class OutputIteratorT,
          class ScanOpT,
          class OffsetT,
          class AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void scan_kernel(
    InputIteratorT d_in,
    AccumT *d_tile_aggregates,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    OffsetT num_items)
{
  // thread block scans tile and writes results to output
  using policy_t                  = typename ChainedPolicyT::ActivePolicy::ReducePolicy;
  constexpr auto items_per_thread = policy_t::BLOCK_THREADS;
  constexpr auto block_threads    = policy_t::ITEMS_PER_THREAD;

  using scan_t            = BlockScan<AccumT, block_threads>;
  using temp_storage_t    = typename scan_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

  constexpr auto tile_items = block_threads * items_per_thread;

  BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);
  prefix_op.running_total = d_tile_aggregates[blockIdx.x];

  OffsetT offset = 0;
  const auto valid_items = CUB_MIN(num_items - offset, tile_items);

  // TODO block load
  AccumT items[items_per_thread];
  LoadDirectBlocked(threadIdx.x, d_in + offset, items, valid_items);

  scan_t(temp_storage).ExclusiveScan(items, items, scan_op, prefix_op);
  CTA_SYNC();

  StoreDirectBlocked(threadIdx.x, d_out + offset, items, valid_items);
}

template <class InputIteratorT,
          class OutputIteratorT,
          class ScanOpT,
          class InitValueT,
          class OffsetT,
          class AccumT = //
          detail::accumulator_t<ScanOpT,
                                cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                                           cub::detail::value_t<InputIteratorT>,
                                                           typename InitValueT::value_type>,
                                cub::detail::value_t<InputIteratorT>>,
          typename SelectedPolicy = DeviceScanPolicy<AccumT, ScanOpT>>
struct dispatch_t : SelectedPolicy
{
  using InputT = cub::detail::value_t<InputIteratorT>;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputIteratorT d_in;
  OutputIteratorT d_out;
  ScanOpT scan_op;
  InitValueT init_value;
  OffsetT num_items;
  cudaStream_t stream;

  int ptx_version;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE dispatch_t(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  template <class ActivePolicyT, class ReduceKernel, class TileScanKernel, class ScanKernel>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  Invoke(ReduceKernel reduce, TileScanKernel tile_scan, ScanKernel scan)
  {
    using policy_t = typename ActivePolicyT::ReducePolicy;

    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    // static_assert(policy_t::LOAD_MODIFIER != CacheLoadModifier::LOAD_LDG,
    //               "The memory consistency model does not apply to texture "
    //               "accesses");

    cudaError error = cudaSuccess;
    do
    {
      // Number of input tiles
      constexpr auto block_threads = policy_t::BLOCK_THREADS;
      constexpr auto tile_size = block_threads * policy_t::ITEMS_PER_THREAD;
      const int num_tiles      = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      allocation_sizes[0] = num_tiles * sizeof(AccumT);

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void* allocations[1] = {};

      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      AccumT* d_tile_aggregates = reinterpret_cast<AccumT*>(allocations[0]);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking reduce_kernel<<<%d, %d, 0, %lld>>>()\n", num_tiles, block_threads, (long long) stream);
#endif
      // Invoke reduce_kernel to get tile aggregates
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_tiles, block_threads, 0, stream)
        .doit(reduce, d_in, d_tile_aggregates, scan_op, num_items);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking tile_scan_kernel<<<1, 1024, 0, %lld>>>()\n", (long long) stream);
#endif
      // Invoke tile_scan_kernel to turn tile aggregates into tile prefixes
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(1, 1024, 0, stream)
        .doit(d_tile_aggregates, scan_op, init_value, num_tiles);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>()\n", num_tiles, block_threads, (long long) stream);
#endif
      // Invoke tile_scan_kernel to turn tile aggregates into tile prefixes
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_tiles, block_threads, 0, stream)
        .doit(d_in, d_tile_aggregates, d_out, scan_op, num_items);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using max_policy_t = typename dispatch_t::MaxPolicy;
    return Invoke<ActivePolicyT>(reduce_kernel<max_policy_t, InputIteratorT, ScanOpT, OffsetT, AccumT>,
                                 tile_scan_kernel<max_policy_t, InitValueT, ScanOpT, AccumT, OffsetT>,
                                 scan_kernel<max_policy_t, InputIteratorT, OutputIteratorT, ScanOpT, OffsetT, AccumT>);
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items,
    cudaStream_t stream)
  {
    typedef typename dispatch_t::MaxPolicy MaxPolicyT;

    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      dispatch_t dispatch(
        d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, scan_op, init_value, stream, ptx_version);

      // Dispatch to chained policy
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

} // namespace run_to_run_deterministic_scan
} // namespace detail

CUB_NAMESPACE_END
