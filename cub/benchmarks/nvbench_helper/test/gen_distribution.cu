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

// Kolmogorov-Smirnov Test
template <class T>
bool is_uniform(thrust::host_vector<T> samples, T min_sample, T max_sample)
{
  if (samples.empty())
  {
    return false;
  }

  if (min_sample >= max_sample)
  {
    return false;
  }

  thrust::sort(samples.begin(), samples.end());

  const auto n = static_cast<double>(samples.size());
  const auto min = static_cast<double>(min_sample);
  const auto max = static_cast<double>(max_sample);

  auto D = 0.0;
  for (std::size_t i = 0; i < samples.size(); i++)
  {
    const T sample = samples[i];

    if (sample < min_sample || sample > max_sample)
    {
      return false;
    }

    const double Fn   = static_cast<double>(i + 1) / n;
    const double F    = (static_cast<double>(sample) - min) / (max - min);
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

// Chi-Square Test for 8-bit data
template <>
bool is_uniform(thrust::host_vector<int8_t> samples, int8_t min_value, int8_t max_value) 
{
  // The specialization is necessary because the Kolmogorov-Smirnov test requires continuous 
  // distributions, and 8-bit data is too descrete so KS test produces a lot of false-positives. 
  // The Chi-Square test is a better fit for discrete distributions. 
  const int k = 255;
  const double expected = samples.size() / static_cast<double>(k);
  std::vector<int> observed(k, 0);
  for (int8_t sample : samples)
  {
    observed[static_cast<unsigned char>(sample - min_value)]++;
  }

  double chi_square = 0.0;
  for (int i = 0; i < k; i++) 
  {
    chi_square += pow(observed[i] - expected, 2) / expected;
  }

  // I don't know how to compute the critical value based on the value range / df, so just use 
  // the constant for df = 255, alpha = 0.05. The test is sensitive to the number of samples and
  // values range, so it's better not to change the parameters. Regardless, it allows us to 
  // catch some bugs in the `int8_t` samples distribution automatically. 
  const double critical = 293.24; 
  return chi_square < critical;
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

TEMPLATE_LIST_TEST_CASE("Generators produce uniformly distributed data", "[gen]", types)
{
  const TestType min = std::numeric_limits<TestType>::min();
  const TestType max = std::numeric_limits<TestType>::max();

  const thrust::device_vector<TestType> data = generate(1 << 16, bit_entropy::_1_000, min, max);

  REQUIRE(is_uniform<TestType>(data, min, max));
}
