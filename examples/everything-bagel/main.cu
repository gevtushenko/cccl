/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This example demonstrates the use of CCCL functionality from Thrust, CUB,
libcu++, and cudax.

The example uses many Thrust algorithms including:
- Searching: find, find_if, find_if_not, mismatch
- Querying: all_of, any_of, none_of, count, count_if
- Sorting queries: is_sorted, is_sorted_until, is_partitioned, partition_point
- Comparisons: equal, mismatch
- Min/Max: min_element, max_element, minmax_element
- Reductions: reduce, transform_reduce, inner_product
- Scans: inclusive_scan, exclusive_scan
- Transformations: transform, replace, replace_if, replace_copy, replace_copy_if
- Copying: copy, copy_if, copy_n
- Filling: fill, fill_n, generate, generate_n, sequence, tabulate
- Removing: remove, remove_if, remove_copy, remove_copy_if, unique, unique_copy
- Reordering: reverse, reverse_copy, shuffle
- Partitioning: partition, partition_copy, stable_partition
- Sorting: sort, stable_sort, sort_by_key, stable_sort_by_key
- Binary search: lower_bound, upper_bound, binary_search, equal_range
- Merging: merge, set_union, set_intersection, set_difference,
set_symmetric_difference
- Gathering/Scattering: gather, scatter
- Adjacent operations: adjacent_difference

CUB device-wide algorithms:
- DeviceReduce: Sum, Min, Max, ArgMin, ArgMax, ReduceByKey
- DeviceScan: ExclusiveSum, InclusiveSum, ExclusiveScan, InclusiveScan
- DeviceSelect: If, Unique, Flagged
- DevicePartition: If, Flagged
- DeviceRadixSort: SortKeys, SortPairs
- DeviceMergeSort: SortKeys
- DeviceRunLengthEncode: Encode
- DeviceHistogram: HistogramEven
- DeviceFor: ForEachN
- DeviceFind: FindIf
- DeviceAdjacentDifference: SubtractLeft
- DeviceMerge: MergeKeys

Plus cudax features for stream management.
*/

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_find.cuh>
#include <cub/device/device_for.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/merge.h>
#include <thrust/mismatch.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <cuda/__functional/maximum.h>

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <iostream>

namespace cudax = cuda::experimental;

struct IsEven {
  __host__ __device__ bool operator()(int x) const { return (x % 2) == 0; }
};

struct IsOdd {
  __host__ __device__ bool operator()(int x) const { return (x % 2) != 0; }
};

struct IsNegative {
  __host__ __device__ bool operator()(int x) const { return x < 0; }
};

struct IsPositive {
  __host__ __device__ bool operator()(int x) const { return x > 0; }
};

struct IsGreaterThan {
  int threshold;

  __host__ __device__ IsGreaterThan(int t) : threshold(t) {}

  __host__ __device__ bool operator()(int x) const { return x > threshold; }
};

struct IsLessThan {
  int threshold;

  __host__ __device__ IsLessThan(int t) : threshold(t) {}

  __host__ __device__ bool operator()(int x) const { return x < threshold; }
};

struct Square {
  __host__ __device__ int operator()(int x) const { return x * x; }
};

struct Negate {
  __host__ __device__ int operator()(int x) const { return -x; }
};

struct Plus {
  __host__ __device__ int operator()(int a, int b) const { return a + b; }
};

struct Multiplies {
  __host__ __device__ int operator()(int a, int b) const { return a * b; }
};

int main() {
  constexpr int N = 1000;

  // Use cuda stream for CUDA stream management
  cuda::stream stream{cuda::devices[0]};

  std::cout << "=== Thrust Algorithm Showcase ===" << std::endl;

  // ==================== INITIALIZATION ====================
  std::cout << "\n--- Initialization ---" << std::endl;

  // thrust::sequence - fill with sequential values
  thrust::device_vector<int> vec1(N);
  thrust::device_vector<int> vec2(N);
  thrust::sequence(vec1.begin(), vec1.end(), 0);
  thrust::sequence(vec2.begin(), vec2.end(), 0);
  std::cout << "thrust::sequence: filled vectors with 0 to " << (N - 1)
            << std::endl;

  // thrust::fill - fill with a constant value
  thrust::device_vector<int> filled(N);
  thrust::fill(filled.begin(), filled.end(), 42);
  std::cout << "thrust::fill: filled vector with 42s" << std::endl;

  // thrust::fill_n - fill first n elements
  thrust::fill_n(filled.begin(), 10, 99);
  std::cout << "thrust::fill_n: filled first 10 elements with 99" << std::endl;

  // thrust::generate - fill using a generator (using tabulate as generate
  // requires host functor)
  thrust::device_vector<int> generated(N);
  thrust::tabulate(generated.begin(), generated.end(), Square{});
  std::cout << "thrust::tabulate: filled with squares (0, 1, 4, 9, ...)"
            << std::endl;

  // ==================== COMPARISON ====================
  std::cout << "\n--- Comparison ---" << std::endl;

  // thrust::equal
  bool are_equal = thrust::equal(vec1.begin(), vec1.end(), vec2.begin());
  std::cout << "thrust::equal: vectors equal = "
            << (are_equal ? "true" : "false") << std::endl;

  // Modify one element
  vec2[500] = -1;

  // thrust::mismatch - find first position where sequences differ
  auto mismatch_pair = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin());
  int mismatch_pos = static_cast<int>(mismatch_pair.first - vec1.begin());
  std::cout << "thrust::mismatch: first difference at index " << mismatch_pos
            << std::endl;

  // Reset vec2
  thrust::sequence(vec2.begin(), vec2.end(), 0);

  // ==================== SEARCHING ====================
  std::cout << "\n--- Searching ---" << std::endl;

  // thrust::find - find first occurrence of a value
  auto find_it = thrust::find(vec1.begin(), vec1.end(), 500);
  int find_pos = static_cast<int>(find_it - vec1.begin());
  std::cout << "thrust::find: value 500 found at index " << find_pos
            << std::endl;

  // thrust::find_if - find first element matching predicate
  auto find_if_it =
      thrust::find_if(vec1.begin(), vec1.end(), IsGreaterThan(900));
  int find_if_pos = static_cast<int>(find_if_it - vec1.begin());
  std::cout << "thrust::find_if: first element > 900 at index " << find_if_pos
            << std::endl;

  // thrust::find_if_not - find first element NOT matching predicate
  auto find_if_not_it =
      thrust::find_if_not(vec1.begin(), vec1.end(), IsLessThan(100));
  int find_if_not_pos = static_cast<int>(find_if_not_it - vec1.begin());
  std::cout << "thrust::find_if_not: first element not < 100 at index "
            << find_if_not_pos << std::endl;

  // ==================== QUERYING ====================
  std::cout << "\n--- Querying (all_of, any_of, none_of, count) ---"
            << std::endl;

  // thrust::all_of - check if all elements satisfy predicate
  bool all_positive = thrust::all_of(vec1.begin(), vec1.end(), IsPositive());
  std::cout << "thrust::all_of: all elements > 0 = "
            << (all_positive ? "true" : "false") << std::endl;

  thrust::device_vector<int> positive_vec(N);
  thrust::sequence(positive_vec.begin(), positive_vec.end(), 1); // 1 to N
  all_positive =
      thrust::all_of(positive_vec.begin(), positive_vec.end(), IsPositive());
  std::cout << "thrust::all_of: all elements in [1,N] > 0 = "
            << (all_positive ? "true" : "false") << std::endl;

  // thrust::any_of - check if any element satisfies predicate
  bool any_greater =
      thrust::any_of(vec1.begin(), vec1.end(), IsGreaterThan(500));
  std::cout << "thrust::any_of: any element > 500 = "
            << (any_greater ? "true" : "false") << std::endl;

  // thrust::none_of - check if no element satisfies predicate
  bool none_negative = thrust::none_of(vec1.begin(), vec1.end(), IsNegative());
  std::cout << "thrust::none_of: no negative elements = "
            << (none_negative ? "true" : "false") << std::endl;

  // thrust::count - count occurrences of a value
  thrust::device_vector<int> with_dupes = {1, 2, 3, 2, 4, 2, 5};
  int count_2 =
      static_cast<int>(thrust::count(with_dupes.begin(), with_dupes.end(), 2));
  std::cout << "thrust::count: count of 2s = " << count_2 << std::endl;

  // thrust::count_if - count elements matching predicate
  int count_even =
      static_cast<int>(thrust::count_if(vec1.begin(), vec1.end(), IsEven()));
  std::cout << "thrust::count_if: count of even numbers = " << count_even
            << std::endl;

  // ==================== SORTING QUERIES ====================
  std::cout << "\n--- Sorting Queries ---" << std::endl;

  // thrust::is_sorted - check if range is sorted
  bool sorted = thrust::is_sorted(vec1.begin(), vec1.end());
  std::cout << "thrust::is_sorted: sequential vector is sorted = "
            << (sorted ? "true" : "false") << std::endl;

  thrust::device_vector<int> unsorted = {3, 1, 4, 1, 5, 9, 2, 6};
  sorted = thrust::is_sorted(unsorted.begin(), unsorted.end());
  std::cout << "thrust::is_sorted: {3,1,4,1,5,9,2,6} is sorted = "
            << (sorted ? "true" : "false") << std::endl;

  // thrust::is_sorted_until - find where sorted order breaks
  auto sorted_until_it =
      thrust::is_sorted_until(unsorted.begin(), unsorted.end());
  int sorted_until_pos = static_cast<int>(sorted_until_it - unsorted.begin());
  std::cout << "thrust::is_sorted_until: sorted until index "
            << sorted_until_pos << std::endl;

  // thrust::is_partitioned - check if range is partitioned
  thrust::device_vector<int> partitioned_vec = {2, 4, 6, 8,
                                                1, 3, 5, 7}; // evens then odds
  bool is_part = thrust::is_partitioned(partitioned_vec.begin(),
                                        partitioned_vec.end(), IsEven());
  std::cout << "thrust::is_partitioned: evens before odds = "
            << (is_part ? "true" : "false") << std::endl;

  // thrust::partition_point - find partition point
  auto partition_pt = thrust::partition_point(partitioned_vec.begin(),
                                              partitioned_vec.end(), IsEven());
  int partition_pos = static_cast<int>(partition_pt - partitioned_vec.begin());
  std::cout << "thrust::partition_point: partition at index " << partition_pos
            << std::endl;

  // ==================== MIN/MAX ====================
  std::cout << "\n--- Min/Max ---" << std::endl;

  thrust::device_vector<int> minmax_vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};

  // thrust::min_element
  auto min_it = thrust::min_element(minmax_vec.begin(), minmax_vec.end());
  int min_pos = static_cast<int>(min_it - minmax_vec.begin());
  int min_val = *min_it;
  std::cout << "thrust::min_element: min = " << min_val << " at index "
            << min_pos << std::endl;

  // thrust::max_element
  auto max_it = thrust::max_element(minmax_vec.begin(), minmax_vec.end());
  int max_pos = static_cast<int>(max_it - minmax_vec.begin());
  int max_val = *max_it;
  std::cout << "thrust::max_element: max = " << max_val << " at index "
            << max_pos << std::endl;

  // thrust::minmax_element
  auto minmax_pair =
      thrust::minmax_element(minmax_vec.begin(), minmax_vec.end());
  std::cout << "thrust::minmax_element: min = " << *minmax_pair.first
            << ", max = " << *minmax_pair.second << std::endl;

  // ==================== REDUCTIONS ====================
  std::cout << "\n--- Reductions ---" << std::endl;

  thrust::device_vector<int> reduce_vec(100);
  thrust::sequence(reduce_vec.begin(), reduce_vec.end(), 1); // 1 to 100

  // thrust::reduce - sum all elements
  int sum = thrust::reduce(reduce_vec.begin(), reduce_vec.end(), 0, Plus());
  std::cout << "thrust::reduce: sum of 1 to 100 = " << sum << std::endl;

  // thrust::transform_reduce - transform then reduce
  int sum_of_squares = thrust::transform_reduce(
      reduce_vec.begin(), reduce_vec.end(), Square(), 0, Plus());
  std::cout << "thrust::transform_reduce: sum of squares = " << sum_of_squares
            << std::endl;

  // thrust::inner_product - dot product
  thrust::device_vector<int> a = {1, 2, 3, 4, 5};
  thrust::device_vector<int> b = {5, 4, 3, 2, 1};
  int dot = thrust::inner_product(a.begin(), a.end(), b.begin(), 0);
  std::cout << "thrust::inner_product: dot product = " << dot << std::endl;

  // ==================== SCANS ====================
  std::cout << "\n--- Scans ---" << std::endl;

  thrust::device_vector<int> scan_input = {1, 2, 3, 4, 5};
  thrust::device_vector<int> scan_output(5);

  // thrust::inclusive_scan
  thrust::inclusive_scan(scan_input.begin(), scan_input.end(),
                         scan_output.begin());
  std::cout << "thrust::inclusive_scan: {1,2,3,4,5} -> {" << scan_output[0]
            << "," << scan_output[1] << "," << scan_output[2] << ","
            << scan_output[3] << "," << scan_output[4] << "}" << std::endl;

  // thrust::exclusive_scan
  thrust::exclusive_scan(scan_input.begin(), scan_input.end(),
                         scan_output.begin(), 0);
  std::cout << "thrust::exclusive_scan: {1,2,3,4,5} -> {" << scan_output[0]
            << "," << scan_output[1] << "," << scan_output[2] << ","
            << scan_output[3] << "," << scan_output[4] << "}" << std::endl;

  // ==================== COPYING ====================
  std::cout << "\n--- Copying ---" << std::endl;

  thrust::device_vector<int> copy_src = {1, 2, 3, 4, 5};
  thrust::device_vector<int> copy_dst(5);

  // thrust::copy
  thrust::copy(copy_src.begin(), copy_src.end(), copy_dst.begin());
  std::cout << "thrust::copy: copied {1,2,3,4,5}" << std::endl;

  // thrust::copy_n
  thrust::device_vector<int> copy_n_dst(3);
  thrust::copy_n(copy_src.begin(), 3, copy_n_dst.begin());
  std::cout << "thrust::copy_n: copied first 3 elements" << std::endl;

  // thrust::copy_if - copy elements matching predicate
  thrust::device_vector<int> mixed = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> evens_only(5);
  auto copy_if_end =
      thrust::copy_if(mixed.begin(), mixed.end(), evens_only.begin(), IsEven());
  int copied_count = static_cast<int>(copy_if_end - evens_only.begin());
  std::cout << "thrust::copy_if: copied " << copied_count << " even numbers"
            << std::endl;

  // ==================== TRANSFORMATIONS ====================
  std::cout << "\n--- Transformations ---" << std::endl;

  thrust::device_vector<int> transform_src = {1, 2, 3, 4, 5};
  thrust::device_vector<int> transform_dst(5);

  // thrust::transform - unary
  thrust::transform(transform_src.begin(), transform_src.end(),
                    transform_dst.begin(), Square());
  std::cout << "thrust::transform (square): {1,2,3,4,5} -> {"
            << transform_dst[0] << "," << transform_dst[1] << ","
            << transform_dst[2] << "," << transform_dst[3] << ","
            << transform_dst[4] << "}" << std::endl;

  // thrust::transform - binary
  thrust::device_vector<int> src_a = {1, 2, 3, 4, 5};
  thrust::device_vector<int> src_b = {10, 20, 30, 40, 50};
  thrust::transform(src_a.begin(), src_a.end(), src_b.begin(),
                    transform_dst.begin(), Plus());
  std::cout << "thrust::transform (add): element-wise sum = {"
            << transform_dst[0] << "," << transform_dst[1] << ","
            << transform_dst[2] << "," << transform_dst[3] << ","
            << transform_dst[4] << "}" << std::endl;

  // thrust::replace
  thrust::device_vector<int> replace_vec = {1, 2, 3, 2, 4, 2, 5};
  thrust::replace(replace_vec.begin(), replace_vec.end(), 2, 99);
  std::cout << "thrust::replace: replaced 2s with 99s" << std::endl;

  // thrust::replace_if
  thrust::device_vector<int> replace_if_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::replace_if(replace_if_vec.begin(), replace_if_vec.end(), IsEven(), 0);
  std::cout << "thrust::replace_if: replaced evens with 0" << std::endl;

  // thrust::replace_copy
  thrust::device_vector<int> replace_copy_src = {1, 2, 3, 2, 4};
  thrust::device_vector<int> replace_copy_dst(5);
  thrust::replace_copy(replace_copy_src.begin(), replace_copy_src.end(),
                       replace_copy_dst.begin(), 2, 99);
  std::cout << "thrust::replace_copy: copied with 2->99" << std::endl;

  // thrust::replace_copy_if
  thrust::device_vector<int> replace_copy_if_src = {1, 2, 3, 4, 5};
  thrust::device_vector<int> replace_copy_if_dst(5);
  thrust::replace_copy_if(replace_copy_if_src.begin(),
                          replace_copy_if_src.end(),
                          replace_copy_if_dst.begin(), IsOdd(), 0);
  std::cout << "thrust::replace_copy_if: copied with odds->0" << std::endl;

  // ==================== REMOVING ====================
  std::cout << "\n--- Removing ---" << std::endl;

  // thrust::remove
  thrust::device_vector<int> remove_vec = {1, 2, 3, 2, 4, 2, 5};
  auto remove_end = thrust::remove(remove_vec.begin(), remove_vec.end(), 2);
  int new_size = static_cast<int>(remove_end - remove_vec.begin());
  std::cout << "thrust::remove: removed 2s, new size = " << new_size
            << std::endl;

  // thrust::remove_if
  thrust::device_vector<int> remove_if_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto remove_if_end =
      thrust::remove_if(remove_if_vec.begin(), remove_if_vec.end(), IsEven());
  new_size = static_cast<int>(remove_if_end - remove_if_vec.begin());
  std::cout << "thrust::remove_if: removed evens, new size = " << new_size
            << std::endl;

  // thrust::remove_copy
  thrust::device_vector<int> remove_copy_src = {1, 2, 3, 2, 4, 2, 5};
  thrust::device_vector<int> remove_copy_dst(4);
  thrust::remove_copy(remove_copy_src.begin(), remove_copy_src.end(),
                      remove_copy_dst.begin(), 2);
  std::cout << "thrust::remove_copy: copied without 2s" << std::endl;

  // thrust::remove_copy_if
  thrust::device_vector<int> remove_copy_if_src = {1, 2, 3, 4, 5,
                                                   6, 7, 8, 9, 10};
  thrust::device_vector<int> remove_copy_if_dst(5);
  thrust::remove_copy_if(remove_copy_if_src.begin(), remove_copy_if_src.end(),
                         remove_copy_if_dst.begin(), IsEven());
  std::cout << "thrust::remove_copy_if: copied without evens" << std::endl;

  // thrust::unique
  thrust::device_vector<int> unique_vec = {1, 1, 2, 2, 2, 3, 3, 4, 5, 5};
  auto unique_end = thrust::unique(unique_vec.begin(), unique_vec.end());
  new_size = static_cast<int>(unique_end - unique_vec.begin());
  std::cout << "thrust::unique: removed consecutive duplicates, new size = "
            << new_size << std::endl;

  // thrust::unique_copy
  thrust::device_vector<int> unique_copy_src = {1, 1, 2, 2, 2, 3, 3, 4, 5, 5};
  thrust::device_vector<int> unique_copy_dst(5);
  auto unique_copy_end = thrust::unique_copy(
      unique_copy_src.begin(), unique_copy_src.end(), unique_copy_dst.begin());
  new_size = static_cast<int>(unique_copy_end - unique_copy_dst.begin());
  std::cout
      << "thrust::unique_copy: copied without consecutive duplicates, size = "
      << new_size << std::endl;

  // ==================== REORDERING ====================
  std::cout << "\n--- Reordering ---" << std::endl;

  // thrust::reverse
  thrust::device_vector<int> reverse_vec = {1, 2, 3, 4, 5};
  thrust::reverse(reverse_vec.begin(), reverse_vec.end());
  std::cout << "thrust::reverse: {1,2,3,4,5} -> {" << reverse_vec[0] << ","
            << reverse_vec[1] << "," << reverse_vec[2] << "," << reverse_vec[3]
            << "," << reverse_vec[4] << "}" << std::endl;

  // thrust::reverse_copy
  thrust::device_vector<int> reverse_copy_src = {1, 2, 3, 4, 5};
  thrust::device_vector<int> reverse_copy_dst(5);
  thrust::reverse_copy(reverse_copy_src.begin(), reverse_copy_src.end(),
                       reverse_copy_dst.begin());
  std::cout << "thrust::reverse_copy: reversed to new vector" << std::endl;

  // thrust::shuffle
  thrust::device_vector<int> shuffle_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::default_random_engine rng(42);
  thrust::shuffle(shuffle_vec.begin(), shuffle_vec.end(), rng);
  std::cout << "thrust::shuffle: shuffled vector" << std::endl;

  // ==================== PARTITIONING ====================
  std::cout << "\n--- Partitioning ---" << std::endl;

  // thrust::partition
  thrust::device_vector<int> partition_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto partition_end =
      thrust::partition(partition_vec.begin(), partition_vec.end(), IsEven());
  int partition_size = static_cast<int>(partition_end - partition_vec.begin());
  std::cout << "thrust::partition: " << partition_size
            << " even numbers moved to front" << std::endl;

  // thrust::stable_partition
  thrust::device_vector<int> stable_partition_vec = {1, 2, 3, 4, 5,
                                                     6, 7, 8, 9, 10};
  thrust::stable_partition(stable_partition_vec.begin(),
                           stable_partition_vec.end(), IsEven());
  std::cout << "thrust::stable_partition: evens before odds, order preserved"
            << std::endl;

  // thrust::partition_copy
  thrust::device_vector<int> partition_copy_src = {1, 2, 3, 4, 5,
                                                   6, 7, 8, 9, 10};
  thrust::device_vector<int> partition_true(5);
  thrust::device_vector<int> partition_false(5);
  thrust::partition_copy(partition_copy_src.begin(), partition_copy_src.end(),
                         partition_true.begin(), partition_false.begin(),
                         IsEven());
  std::cout << "thrust::partition_copy: split into evens and odds" << std::endl;

  // ==================== SORTING ====================
  std::cout << "\n--- Sorting ---" << std::endl;

  // thrust::sort
  thrust::device_vector<int> sort_vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
  thrust::sort(sort_vec.begin(), sort_vec.end());
  std::cout << "thrust::sort: sorted ascending" << std::endl;

  // thrust::stable_sort
  thrust::device_vector<int> stable_sort_vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
  thrust::stable_sort(stable_sort_vec.begin(), stable_sort_vec.end());
  std::cout << "thrust::stable_sort: stable sorted ascending" << std::endl;

  // thrust::sort_by_key
  thrust::device_vector<int> keys = {5, 2, 8, 1, 9};
  thrust::device_vector<int> values = {50, 20, 80, 10, 90};
  thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
  std::cout << "thrust::sort_by_key: sorted keys with corresponding values"
            << std::endl;

  // thrust::stable_sort_by_key
  thrust::device_vector<int> stable_keys = {5, 2, 8, 1, 9};
  thrust::device_vector<int> stable_values = {50, 20, 80, 10, 90};
  thrust::stable_sort_by_key(stable_keys.begin(), stable_keys.end(),
                             stable_values.begin());
  std::cout << "thrust::stable_sort_by_key: stable sorted keys with values"
            << std::endl;

  // ==================== BINARY SEARCH ====================
  std::cout << "\n--- Binary Search ---" << std::endl;

  thrust::device_vector<int> sorted_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // thrust::lower_bound
  auto lower = thrust::lower_bound(sorted_vec.begin(), sorted_vec.end(), 5);
  int lower_pos = static_cast<int>(lower - sorted_vec.begin());
  std::cout << "thrust::lower_bound: lower bound of 5 at index " << lower_pos
            << std::endl;

  // thrust::upper_bound
  auto upper = thrust::upper_bound(sorted_vec.begin(), sorted_vec.end(), 5);
  int upper_pos = static_cast<int>(upper - sorted_vec.begin());
  std::cout << "thrust::upper_bound: upper bound of 5 at index " << upper_pos
            << std::endl;

  // thrust::binary_search
  bool found = thrust::binary_search(sorted_vec.begin(), sorted_vec.end(), 5);
  std::cout << "thrust::binary_search: 5 found = " << (found ? "true" : "false")
            << std::endl;

  // thrust::equal_range
  auto range = thrust::equal_range(sorted_vec.begin(), sorted_vec.end(), 5);
  int range_begin = static_cast<int>(range.first - sorted_vec.begin());
  int range_end = static_cast<int>(range.second - sorted_vec.begin());
  std::cout << "thrust::equal_range: range for 5 = [" << range_begin << ", "
            << range_end << ")" << std::endl;

  // ==================== MERGING ====================
  std::cout << "\n--- Merging ---" << std::endl;

  // thrust::merge
  thrust::device_vector<int> merge_a = {1, 3, 5, 7, 9};
  thrust::device_vector<int> merge_b = {2, 4, 6, 8, 10};
  thrust::device_vector<int> merged(10);
  thrust::merge(merge_a.begin(), merge_a.end(), merge_b.begin(), merge_b.end(),
                merged.begin());
  std::cout << "thrust::merge: merged two sorted sequences" << std::endl;

  // ==================== SET OPERATIONS ====================
  std::cout << "\n--- Set Operations ---" << std::endl;

  thrust::device_vector<int> set_a = {1, 2, 3, 4, 5};
  thrust::device_vector<int> set_b = {3, 4, 5, 6, 7};
  thrust::device_vector<int> set_result(10);

  // thrust::set_union
  auto set_union_end =
      thrust::set_union(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
                        set_result.begin());
  int set_union_size = static_cast<int>(set_union_end - set_result.begin());
  std::cout << "thrust::set_union: union size = " << set_union_size
            << std::endl;

  // thrust::set_intersection
  auto set_intersection_end =
      thrust::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
                               set_b.end(), set_result.begin());
  int set_intersection_size =
      static_cast<int>(set_intersection_end - set_result.begin());
  std::cout << "thrust::set_intersection: intersection size = "
            << set_intersection_size << std::endl;

  // thrust::set_difference
  auto set_diff_end =
      thrust::set_difference(set_a.begin(), set_a.end(), set_b.begin(),
                             set_b.end(), set_result.begin());
  int set_diff_size = static_cast<int>(set_diff_end - set_result.begin());
  std::cout << "thrust::set_difference: difference size = " << set_diff_size
            << std::endl;

  // thrust::set_symmetric_difference
  auto set_sym_diff_end = thrust::set_symmetric_difference(
      set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
      set_result.begin());
  int set_sym_diff_size =
      static_cast<int>(set_sym_diff_end - set_result.begin());
  std::cout << "thrust::set_symmetric_difference: symmetric diff size = "
            << set_sym_diff_size << std::endl;

  // ==================== GATHER/SCATTER ====================
  std::cout << "\n--- Gather/Scatter ---" << std::endl;

  // thrust::gather
  thrust::device_vector<int> gather_map = {3, 1, 4, 1, 5};
  thrust::device_vector<int> gather_src = {0, 10, 20, 30, 40, 50, 60};
  thrust::device_vector<int> gather_result(5);
  thrust::gather(gather_map.begin(), gather_map.end(), gather_src.begin(),
                 gather_result.begin());
  std::cout << "thrust::gather: gathered elements at indices {3,1,4,1,5}"
            << std::endl;

  // thrust::scatter
  thrust::device_vector<int> scatter_src = {10, 20, 30, 40, 50};
  thrust::device_vector<int> scatter_map = {2, 0, 4, 1, 3};
  thrust::device_vector<int> scatter_dst(5, 0);
  thrust::scatter(scatter_src.begin(), scatter_src.end(), scatter_map.begin(),
                  scatter_dst.begin());
  std::cout << "thrust::scatter: scattered to indices {2,0,4,1,3}" << std::endl;

  // ==================== ADJACENT OPERATIONS ====================
  std::cout << "\n--- Adjacent Difference ---" << std::endl;

  // thrust::adjacent_difference
  thrust::device_vector<int> adj_diff_src = {1, 4, 9, 16, 25};
  thrust::device_vector<int> adj_diff_dst(5);
  thrust::adjacent_difference(adj_diff_src.begin(), adj_diff_src.end(),
                              adj_diff_dst.begin());
  std::cout << "thrust::adjacent_difference: {1,4,9,16,25} -> {"
            << adj_diff_dst[0] << "," << adj_diff_dst[1] << ","
            << adj_diff_dst[2] << "," << adj_diff_dst[3] << ","
            << adj_diff_dst[4] << "}" << std::endl;

  // ==================== CUB DEVICE ALGORITHMS ====================
  std::cout << "\n=== CUB Device Algorithm Showcase ===" << std::endl;

  // Helper lambda to allocate temp storage and run CUB algorithms
  auto run_cub_algorithm = [&stream](auto algorithm_fn) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call to get temp storage size
    cudaError_t err = algorithm_fn(d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
      std::cerr << "Error determining temp storage: " << cudaGetErrorString(err)
                << std::endl;
      return err;
    }

    // Allocate temp storage
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Run the algorithm
    err = algorithm_fn(d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
      std::cerr << "Error running algorithm: " << cudaGetErrorString(err)
                << std::endl;
    }
    return err;
  };

  // ==================== CUB DeviceReduce ====================
  std::cout << "\n--- CUB DeviceReduce ---" << std::endl;

  thrust::device_vector<int> reduce_input(N);
  thrust::sequence(reduce_input.begin(), reduce_input.end(), 1); // 1 to N
  thrust::device_vector<int> reduce_output(1);

  // DeviceReduce::Sum
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::Sum(
        temp, temp_bytes, thrust::raw_pointer_cast(reduce_input.data()),
        thrust::raw_pointer_cast(reduce_output.data()), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::Sum: sum of 1 to " << N << " = "
            << reduce_output[0] << std::endl;

  // DeviceReduce::Min
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::Min(
        temp, temp_bytes, thrust::raw_pointer_cast(reduce_input.data()),
        thrust::raw_pointer_cast(reduce_output.data()), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::Min: min = " << reduce_output[0]
            << std::endl;

  // DeviceReduce::Max
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::Max(
        temp, temp_bytes, thrust::raw_pointer_cast(reduce_input.data()),
        thrust::raw_pointer_cast(reduce_output.data()), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::Max: max = " << reduce_output[0]
            << std::endl;

  // DeviceReduce::ArgMin (new API with separate value and index outputs)
  thrust::device_vector<int> argmin_value_out(1);
  thrust::device_vector<int64_t> argmin_index_out(1);
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::ArgMin(
        temp, temp_bytes, thrust::raw_pointer_cast(reduce_input.data()),
        thrust::raw_pointer_cast(argmin_value_out.data()),
        thrust::raw_pointer_cast(argmin_index_out.data()), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::ArgMin: min value " << argmin_value_out[0]
            << " at index " << argmin_index_out[0] << std::endl;

  // DeviceReduce::ArgMax (new API with separate value and index outputs)
  thrust::device_vector<int> argmax_value_out(1);
  thrust::device_vector<int64_t> argmax_index_out(1);
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::ArgMax(
        temp, temp_bytes, thrust::raw_pointer_cast(reduce_input.data()),
        thrust::raw_pointer_cast(argmax_value_out.data()),
        thrust::raw_pointer_cast(argmax_index_out.data()), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::ArgMax: max value " << argmax_value_out[0]
            << " at index " << argmax_index_out[0] << std::endl;

  // DeviceReduce::ReduceByKey
  thrust::device_vector<int> rbk_keys_in = {0, 0, 0, 1, 1, 2, 2, 2, 2, 3};
  thrust::device_vector<int> rbk_values_in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> rbk_keys_out(4);
  thrust::device_vector<int> rbk_values_out(4);
  thrust::device_vector<int> rbk_num_runs(1);
  int rbk_n = static_cast<int>(rbk_keys_in.size());

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceReduce::ReduceByKey(
        temp, temp_bytes, thrust::raw_pointer_cast(rbk_keys_in.data()),
        thrust::raw_pointer_cast(rbk_keys_out.data()),
        thrust::raw_pointer_cast(rbk_values_in.data()),
        thrust::raw_pointer_cast(rbk_values_out.data()),
        thrust::raw_pointer_cast(rbk_num_runs.data()), ::cuda::std::plus<>{},
        rbk_n, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceReduce::ReduceByKey: " << rbk_num_runs[0]
            << " unique keys, sums = {" << rbk_values_out[0] << ","
            << rbk_values_out[1] << "," << rbk_values_out[2] << ","
            << rbk_values_out[3] << "}" << std::endl;

  // ==================== CUB DeviceScan ====================
  std::cout << "\n--- CUB DeviceScan ---" << std::endl;

  thrust::device_vector<int> cub_scan_input(10);
  thrust::sequence(cub_scan_input.begin(), cub_scan_input.end(), 1); // 1 to 10
  thrust::device_vector<int> cub_scan_output(10);

  // DeviceScan::ExclusiveSum
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceScan::ExclusiveSum(
        temp, temp_bytes, thrust::raw_pointer_cast(cub_scan_input.data()),
        thrust::raw_pointer_cast(cub_scan_output.data()), 10, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceScan::ExclusiveSum: {1..10} -> {"
            << cub_scan_output[0] << "," << cub_scan_output[1] << ","
            << cub_scan_output[2] << ",...," << cub_scan_output[9] << "}"
            << std::endl;

  // DeviceScan::InclusiveSum
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceScan::InclusiveSum(
        temp, temp_bytes, thrust::raw_pointer_cast(cub_scan_input.data()),
        thrust::raw_pointer_cast(cub_scan_output.data()), 10, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceScan::InclusiveSum: {1..10} -> {"
            << cub_scan_output[0] << "," << cub_scan_output[1] << ","
            << cub_scan_output[2] << ",...," << cub_scan_output[9] << "}"
            << std::endl;

  // DeviceScan::ExclusiveScan with custom op
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceScan::ExclusiveScan(
        temp, temp_bytes, thrust::raw_pointer_cast(cub_scan_input.data()),
        thrust::raw_pointer_cast(cub_scan_output.data()),
        ::cuda::maximum<int>{},
        0, // init value
        10, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceScan::ExclusiveScan (max): {1..10} -> {"
            << cub_scan_output[0] << "," << cub_scan_output[1] << ","
            << cub_scan_output[2] << ",...," << cub_scan_output[9] << "}"
            << std::endl;

  // DeviceScan::InclusiveScan with custom op
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceScan::InclusiveScan(
        temp, temp_bytes, thrust::raw_pointer_cast(cub_scan_input.data()),
        thrust::raw_pointer_cast(cub_scan_output.data()),
        ::cuda::maximum<int>{}, 10, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceScan::InclusiveScan (max): {1..10} -> {"
            << cub_scan_output[0] << "," << cub_scan_output[1] << ","
            << cub_scan_output[2] << ",...," << cub_scan_output[9] << "}"
            << std::endl;

  // ==================== CUB DeviceSelect ====================
  std::cout << "\n--- CUB DeviceSelect ---" << std::endl;

  thrust::device_vector<int> select_input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> select_output(10);
  thrust::device_vector<int> num_selected(1);

  // DeviceSelect::If - select even numbers
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceSelect::If(temp, temp_bytes,
                                 thrust::raw_pointer_cast(select_input.data()),
                                 thrust::raw_pointer_cast(select_output.data()),
                                 thrust::raw_pointer_cast(num_selected.data()),
                                 10, IsEven(), stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceSelect::If (evens): selected " << num_selected[0]
            << " elements: {" << select_output[0] << "," << select_output[1]
            << "," << select_output[2] << "," << select_output[3] << ","
            << select_output[4] << "}" << std::endl;

  // DeviceSelect::Unique
  thrust::device_vector<int> unique_input = {1, 1, 2, 2, 2, 3, 3, 4, 5, 5};
  thrust::device_vector<int> unique_output(10);
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceSelect::Unique(
        temp, temp_bytes, thrust::raw_pointer_cast(unique_input.data()),
        thrust::raw_pointer_cast(unique_output.data()),
        thrust::raw_pointer_cast(num_selected.data()), 10, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceSelect::Unique: " << num_selected[0]
            << " unique elements" << std::endl;

  // DeviceSelect::Flagged
  thrust::device_vector<int> flagged_input = {1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> flags = {1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> flagged_output(8);
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceSelect::Flagged(
        temp, temp_bytes, thrust::raw_pointer_cast(flagged_input.data()),
        thrust::raw_pointer_cast(flags.data()),
        thrust::raw_pointer_cast(flagged_output.data()),
        thrust::raw_pointer_cast(num_selected.data()), 8, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceSelect::Flagged: selected " << num_selected[0]
            << " flagged elements" << std::endl;

  // ==================== CUB DevicePartition ====================
  std::cout << "\n--- CUB DevicePartition ---" << std::endl;

  thrust::device_vector<int> partition_input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> partition_output(10);
  thrust::device_vector<int> num_selected_partition(1);

  // DevicePartition::If - partition evens to front
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DevicePartition::If(
        temp, temp_bytes, thrust::raw_pointer_cast(partition_input.data()),
        thrust::raw_pointer_cast(partition_output.data()),
        thrust::raw_pointer_cast(num_selected_partition.data()), 10, IsEven(),
        stream.get());
  });
  stream.sync();
  std::cout << "cub::DevicePartition::If: " << num_selected_partition[0]
            << " evens at front" << std::endl;

  // DevicePartition::Flagged
  thrust::device_vector<int> part_flags = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DevicePartition::Flagged(
        temp, temp_bytes, thrust::raw_pointer_cast(partition_input.data()),
        thrust::raw_pointer_cast(part_flags.data()),
        thrust::raw_pointer_cast(partition_output.data()),
        thrust::raw_pointer_cast(num_selected_partition.data()), 10,
        stream.get());
  });
  stream.sync();
  std::cout << "cub::DevicePartition::Flagged: " << num_selected_partition[0]
            << " flagged at front" << std::endl;

  // ==================== CUB DeviceRadixSort ====================
  std::cout << "\n--- CUB DeviceRadixSort ---" << std::endl;

  thrust::device_vector<int> sort_keys_in = {8, 3, 7, 1, 9, 2, 6, 4, 5, 0};
  thrust::device_vector<int> sort_keys_out(10);

  // DeviceRadixSort::SortKeys
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceRadixSort::SortKeys(
        temp, temp_bytes, thrust::raw_pointer_cast(sort_keys_in.data()),
        thrust::raw_pointer_cast(sort_keys_out.data()), 10, 0, sizeof(int) * 8,
        stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceRadixSort::SortKeys: sorted to {" << sort_keys_out[0]
            << "," << sort_keys_out[1] << "," << sort_keys_out[2] << ",...,"
            << sort_keys_out[9] << "}" << std::endl;

  // DeviceRadixSort::SortKeysDescending
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceRadixSort::SortKeysDescending(
        temp, temp_bytes, thrust::raw_pointer_cast(sort_keys_in.data()),
        thrust::raw_pointer_cast(sort_keys_out.data()), 10, 0, sizeof(int) * 8,
        stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceRadixSort::SortKeysDescending: sorted to {"
            << sort_keys_out[0] << "," << sort_keys_out[1] << ","
            << sort_keys_out[2] << ",...," << sort_keys_out[9] << "}"
            << std::endl;

  // DeviceRadixSort::SortPairs
  thrust::device_vector<int> sort_values_in = {80, 30, 70, 10, 90,
                                               20, 60, 40, 50, 0};
  thrust::device_vector<int> sort_values_out(10);
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceRadixSort::SortPairs(
        temp, temp_bytes, thrust::raw_pointer_cast(sort_keys_in.data()),
        thrust::raw_pointer_cast(sort_keys_out.data()),
        thrust::raw_pointer_cast(sort_values_in.data()),
        thrust::raw_pointer_cast(sort_values_out.data()), 10, 0,
        sizeof(int) * 8, stream.get());
  });
  stream.sync();
  std::cout
      << "cub::DeviceRadixSort::SortPairs: keys and values sorted together"
      << std::endl;

  // ==================== CUB DeviceMergeSort ====================
  std::cout << "\n--- CUB DeviceMergeSort ---" << std::endl;

  thrust::device_vector<int> merge_sort_data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};

  // DeviceMergeSort::SortKeys
  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceMergeSort::SortKeys(
        temp, temp_bytes, thrust::raw_pointer_cast(merge_sort_data.data()), 10,
        ::cuda::std::less<>{}, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceMergeSort::SortKeys: sorted to {"
            << merge_sort_data[0] << "," << merge_sort_data[1] << ","
            << merge_sort_data[2] << ",...," << merge_sort_data[9] << "}"
            << std::endl;

  // ==================== CUB DeviceRunLengthEncode ====================
  std::cout << "\n--- CUB DeviceRunLengthEncode ---" << std::endl;

  thrust::device_vector<int> rle_input = {0, 2, 2, 9, 5, 5, 5, 8};
  thrust::device_vector<int> rle_unique(8);
  thrust::device_vector<int> rle_counts(8);
  thrust::device_vector<int> rle_num_runs(1);

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceRunLengthEncode::Encode(
        temp, temp_bytes, thrust::raw_pointer_cast(rle_input.data()),
        thrust::raw_pointer_cast(rle_unique.data()),
        thrust::raw_pointer_cast(rle_counts.data()),
        thrust::raw_pointer_cast(rle_num_runs.data()), 8, stream.get());
  });
  stream.sync();
  int num_runs = rle_num_runs[0];
  std::cout << "cub::DeviceRunLengthEncode::Encode: " << num_runs
            << " runs, values = {" << rle_unique[0] << "," << rle_unique[1]
            << "," << rle_unique[2] << "," << rle_unique[3] << ","
            << rle_unique[4] << "}, counts = {" << rle_counts[0] << ","
            << rle_counts[1] << "," << rle_counts[2] << "," << rle_counts[3]
            << "," << rle_counts[4] << "}" << std::endl;

  // ==================== CUB DeviceHistogram ====================
  std::cout << "\n--- CUB DeviceHistogram ---" << std::endl;

  thrust::device_vector<int> hist_samples = {0, 1, 1, 2, 2, 2, 3, 3,
                                             3, 3, 4, 4, 4, 4, 4};
  thrust::device_vector<int> histogram(5, 0);

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceHistogram::HistogramEven(
        temp, temp_bytes, thrust::raw_pointer_cast(hist_samples.data()),
        thrust::raw_pointer_cast(histogram.data()),
        6, // num_levels (5 bins)
        0, // lower_level
        5, // upper_level
        15, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceHistogram::HistogramEven: histogram = {"
            << histogram[0] << "," << histogram[1] << "," << histogram[2] << ","
            << histogram[3] << "," << histogram[4] << "}" << std::endl;

  // ==================== CUB DeviceFor ====================
  std::cout << "\n--- CUB DeviceFor ---" << std::endl;

  thrust::device_vector<int> for_data(10, 1);
  auto double_op = [] __device__(int &x) { x *= 2; };

  cudaError_t err = cub::DeviceFor::ForEachN(
      thrust::raw_pointer_cast(for_data.data()), 10, double_op, stream.get());
  if (err != cudaSuccess) {
    std::cerr << "Error in ForEachN: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  stream.sync();
  std::cout << "cub::DeviceFor::ForEachN: doubled values, first = "
            << for_data[0] << std::endl;

  // ==================== CUB DeviceFind ====================
  std::cout << "\n--- CUB DeviceFind ---" << std::endl;

  thrust::device_vector<int> find_data(N);
  thrust::device_vector<int> find_result(1);
  thrust::sequence(find_data.begin(), find_data.end(), 0);

  int threshold = 500;

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceFind::FindIf(temp, temp_bytes,
                                   thrust::raw_pointer_cast(find_data.data()),
                                   thrust::raw_pointer_cast(find_result.data()),
                                   IsGreaterThan(threshold), N, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceFind::FindIf: first element > " << threshold
            << " at index " << find_result[0] << std::endl;

  // ==================== CUB DeviceAdjacentDifference ====================
  std::cout << "\n--- CUB DeviceAdjacentDifference ---" << std::endl;

  thrust::device_vector<int> adj_input = {1, 4, 9, 16, 25, 36, 49, 64};
  thrust::device_vector<int> adj_output(8);

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceAdjacentDifference::SubtractLeftCopy(
        temp, temp_bytes, thrust::raw_pointer_cast(adj_input.data()),
        thrust::raw_pointer_cast(adj_output.data()), 8, ::cuda::std::minus<>{},
        stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceAdjacentDifference::SubtractLeft: "
               "{1,4,9,16,25,36,49,64} -> {"
            << adj_output[0] << "," << adj_output[1] << "," << adj_output[2]
            << "," << adj_output[3] << "," << adj_output[4] << ","
            << adj_output[5] << "," << adj_output[6] << "," << adj_output[7]
            << "}" << std::endl;

  // ==================== CUB DeviceMerge ====================
  std::cout << "\n--- CUB DeviceMerge ---" << std::endl;

  thrust::device_vector<int> cub_merge_a = {1, 3, 5, 7, 9};
  thrust::device_vector<int> cub_merge_b = {2, 4, 6, 8, 10};
  thrust::device_vector<int> cub_merge_output(10);

  run_cub_algorithm([&](void *temp, size_t &temp_bytes) {
    return cub::DeviceMerge::MergeKeys(
        temp, temp_bytes, thrust::raw_pointer_cast(cub_merge_a.data()), 5,
        thrust::raw_pointer_cast(cub_merge_b.data()), 5,
        thrust::raw_pointer_cast(cub_merge_output.data()),
        ::cuda::std::less<>{}, stream.get());
  });
  stream.sync();
  std::cout << "cub::DeviceMerge::MergeKeys: merged to {" << cub_merge_output[0]
            << "," << cub_merge_output[1] << "," << cub_merge_output[2]
            << ",...," << cub_merge_output[9] << "}" << std::endl;

  std::cout << "\n=== All tests passed! ===" << std::endl;
  return 0;
}
