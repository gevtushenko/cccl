#include <cub/device/device_run_length_encode.cuh>

#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>

#include <catch2/catch.hpp>
#include <nvbench_helper.cuh>

template <class T>
double get_expected_entropy(bit_entropy in_entropy)
{
  if (in_entropy == bit_entropy::_0_000)
  {
    return 0.0;
  }

  if (in_entropy == bit_entropy::_1_000)
  {
    return sizeof(T) * 8;
  }

  const int samples    = static_cast<int>(in_entropy) + 1;
  const double p1      = std::pow(0.5, samples);
  const double p2      = 1 - p1;
  const double entropy = (-p1 * std::log2(p1)) + (-p2 * std::log2(p2));
  return sizeof(T) * 8 * entropy;
}

template <class T>
double compute_actual_entropy(thrust::device_vector<T> in)
{
  const int n = static_cast<int>(in.size());
  thrust::device_vector<T> unique(n);
  thrust::device_vector<int> counts(n);
  thrust::device_vector<int> num_runs(1);

  thrust::sort(in.begin(), in.end());

  // RLE
  void *d_temp_storage           = nullptr;
  std::size_t temp_storage_bytes = 0;

  T *d_in             = thrust::raw_pointer_cast(in.data());
  T *d_unique_out     = thrust::raw_pointer_cast(unique.data());
  int *d_counts_out   = thrust::raw_pointer_cast(counts.data());
  int *d_num_runs_out = thrust::raw_pointer_cast(num_runs.data());

  cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_unique_out,
                                     d_counts_out,
                                     d_num_runs_out,
                                     n);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_unique_out,
                                     d_counts_out,
                                     d_num_runs_out,
                                     n);

  thrust::host_vector<int> h_counts   = counts;
  thrust::host_vector<int> h_num_runs = num_runs;

  // normalize counts
  thrust::host_vector<double> ps(h_num_runs[0]);
  for (std::size_t i = 0; i < ps.size(); i++)
  {
    ps[i] = static_cast<double>(h_counts[i]) / n;
  }

  double entropy = 0.0;

  if (ps.size())
  {
    for (double p : ps)
    {
      entropy -= p * std::log2(p);
    }
  }

  return entropy;
}

TEMPLATE_LIST_TEST_CASE("Generators produce data with given entropy", "[gen]", fundamental_types)
{
  constexpr int num_entropy_levels = 6;
  std::array<bit_entropy, num_entropy_levels> entropy_levels{bit_entropy::_0_000,
                                                             bit_entropy::_0_201,
                                                             bit_entropy::_0_337,
                                                             bit_entropy::_0_544,
                                                             bit_entropy::_0_811,
                                                             bit_entropy::_1_000};

  std::vector<double> entropy(num_entropy_levels);
  std::transform(entropy_levels.cbegin(),
                 entropy_levels.cend(),
                 entropy.begin(),
                 [](bit_entropy entropy) {
                   const thrust::device_vector<TestType> data = generate(1 << 24, entropy);
                   return compute_actual_entropy(data);
                 });

  REQUIRE(std::is_sorted(entropy.begin(), entropy.end()));
  REQUIRE(std::unique(entropy.begin(), entropy.end()) == entropy.end());
}
