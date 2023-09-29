/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <limits>

#include <catch2/catch.hpp>
#include <nvbench_helper.cuh>

bool is_uniform(thrust::host_vector<double> samples,
                double min = std::numeric_limits<double>::min(),
                double max = std::numeric_limits<double>::max())
{
  if (samples.empty())
  {
    return false;
  }
  if (min >= max)
  {
    return false;
  }
  thrust::host_vector<double> sorted_samples = samples;
  thrust::sort(sorted_samples.begin(), sorted_samples.end());
  double n = sorted_samples.size();
  double D = 0.0;
  for (std::size_t i = 0; i < sorted_samples.size(); i++)
  {
    const double sample = sorted_samples[i];
    if (sample < min || sample > max)
    {
      return false;
    }
    // Empirical uniform distribution function
    const double Fn   = (i + 1) / n;
    const double F    = (sample - min) / (max - min);
    const double diff = std::abs(F - Fn);
    if (diff > D)
    {
      D = diff;
    }
  }
  const double c_alpha        = 1.36; // constant for n > 50, alpha = 0.05
  const double critical_value = c_alpha / std::sqrt(n);
  return D <= critical_value;
}

TEST_CASE("Generators produce uniformly distributed data", "[gen]")
{
  const thrust::device_vector<double> data = generate(1 << 24);

  REQUIRE(is_uniform(data));
}
