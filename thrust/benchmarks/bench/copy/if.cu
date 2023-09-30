#include <nvbench_helper.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

template <class T>
struct less_then_t 
{
  T m_val;

  __device__ bool operator()(const T &val) const { return val < m_val; }
};

template <typename T>
T value_from_entropy(double percentage) 
{
  if (percentage == 1) 
  {
    return std::numeric_limits<T>::max();
  }
  
  const auto max_val = static_cast<double>(std::numeric_limits<T>::max());
  const auto min_val = static_cast<double>(std::numeric_limits<T>::lowest());
  const auto result = min_val + percentage * max_val - percentage * min_val;
  return static_cast<T>(result);
}

template <typename T>
static void basic(nvbench::state &state,
                  nvbench::type_list<T>)
{
  using select_op_t = less_then_t<T>;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  T val = value_from_entropy<T>(entropy_to_probability(entropy));
  select_op_t select_op{val};

  thrust::device_vector<T> input = generate(elements);
  const auto selected_elements = thrust::count_if(input.cbegin(), input.cend(), select_op);
  thrust::device_vector<T> output(selected_elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(selected_elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    thrust::copy_if(input.cbegin(), input.cend(), output.begin(), select_op);
  });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::copy_if")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
