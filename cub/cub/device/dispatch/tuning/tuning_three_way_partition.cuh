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

#pragma once

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/config.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace three_way_partition
{

enum class input_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class InputT>
constexpr input_size classify_input_size()
{
  return sizeof(InputT) == 1    ? input_size::_1
         : sizeof(InputT) == 2  ? input_size::_2
         : sizeof(InputT) == 4  ? input_size::_4
         : sizeof(InputT) == 8  ? input_size::_8
         : sizeof(InputT) == 16 ? input_size::_16
                                : input_size::unknown;
}

template <class InputT, input_size InputSize = classify_input_size<InputT>()>
struct sm90_tuning
{
  static constexpr int threads = 256;

  static constexpr int items = Nominal4BItemsToItems<InputT>(9);

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::fixed_delay_constructor_t<350, 450>;
};

template <class Input>
struct sm90_tuning<Input, input_size::_1>
{
  static constexpr int threads = 384;
  static constexpr int items   = 15;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<140>;
};

template <class Input>
struct sm90_tuning<Input, input_size::_2>
{
  static constexpr int threads = 448;
  static constexpr int items   = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  using delay_constructor = detail::no_delay_constructor_t<990>;
};

template <class Input>
struct sm90_tuning<Input, input_size::_4>
{
  static constexpr int threads = 448;
  static constexpr int items   = 13;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<680>;
};

template <class Input>
struct sm90_tuning<Input, input_size::_8>
{
  static constexpr int threads = 448;
  static constexpr int items   = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_TRANSPOSE;

  using delay_constructor = detail::no_delay_constructor_t<1055>;
};

template <class Input>
struct sm90_tuning<Input, input_size::_16>
{
  static constexpr int threads = 128;
  static constexpr int items   = 21;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_TRANSPOSE;

  using delay_constructor = detail::fixed_delay_constructor_t<292, 1150>;
};

} // namespace three_way_partition

template <class InputT>
struct device_three_way_partition_policy_hub
{
  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    constexpr static int ITEMS_PER_THREAD = Nominal4BItemsToItems<InputT>(9);

    using ThreeWayPartitionPolicy = cub::AgentThreeWayPartitionPolicy<256,
                                                                      ITEMS_PER_THREAD,
                                                                      cub::BLOCK_LOAD_DIRECT,
                                                                      cub::LOAD_DEFAULT,
                                                                      cub::BLOCK_SCAN_WARP_SCANS>;
  };

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy350>
  {
    using tuning = detail::three_way_partition::sm90_tuning<InputT>;

    using ThreeWayPartitionPolicy =
      AgentThreeWayPartitionPolicy<tuning::threads,
                                   tuning::items,
                                   tuning::load_algorithm,
                                   cub::LOAD_DEFAULT,
                                   cub::BLOCK_SCAN_WARP_SCANS,
                                   typename tuning::delay_constructor>;
  };

  using MaxPolicy = Policy900;
};

} // namespace detail

CUB_NAMESPACE_END
