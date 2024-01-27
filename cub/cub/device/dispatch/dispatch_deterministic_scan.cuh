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

#include <cub/agent/agent_scan.cuh>
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

template <class ChainedPolicyT,
          class InputIteratorT,
          class ScanOpT,
          class InitValueT,
          class OffsetT,
          class AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void reduce_kernel(
    InputIteratorT d_in,
    AccumT *d_tile_aggregates,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items)
{
  // thread block reduces tile to one aggregate
}

template <class ChainedPolicyT, class ScanOpT, class AccumT, class OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
CUB_DETAIL_KERNEL_ATTRIBUTES void tile_scan_kernel(AccumT *d_tile_aggregates, ScanOpT scan_op, OffsetT num_tiles)
{
  // single thread block computes prefix sum of tile aggregates
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
    AccumT d_tile_aggregates,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    OffsetT num_items)
{
  // thread block scans tile and writes results to output
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT = //
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

  template <typename ActivePolicyT, typename ReduceKernel, typename ScanKernel>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  Invoke(ReduceKernel reduce_kernel, ScanKernel scan_kernel)
  {
    typedef typename ActivePolicyT::ScanPolicyT Policy;

    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    static_assert(Policy::LOAD_MODIFIER != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");

    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
      int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

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

      if (d_temp_storage == NULL)
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

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(init_kernel, tile_state, num_tiles);

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

      // Get SM occupancy for scan_kernel
      int scan_sm_occupancy;
      error = CubDebug(MaxSmOccupancy(scan_sm_occupancy, // out
                                      scan_kernel,
                                      Policy::BLOCK_THREADS));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
      {
// Log scan_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items "
                "per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                Policy::BLOCK_THREADS,
                (long long) stream,
                Policy::ITEMS_PER_THREAD,
                scan_sm_occupancy);
#endif

        // Invoke scan_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(scan_grid_size, Policy::BLOCK_THREADS, 0, stream)
          .doit(scan_kernel, d_in, d_out, tile_state, start_tile, scan_op, init_value, num_items);

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
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    typedef typename dispatch_t::MaxPolicy MaxPolicyT;
    typedef typename cub::ScanTileState<AccumT> ScanTileStateT;
    // Ensure kernels are instantiated.
    return Invoke<ActivePolicyT>(
      reduce_kernel<ScanTileStateT>,
      scan_kernel<MaxPolicyT, InputIteratorT, OutputIteratorT, ScanTileStateT, ScanOpT, InitValueT, OffsetT, AccumT>);
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
