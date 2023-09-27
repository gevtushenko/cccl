if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set to build Thrust benchmarks.")
endif()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach(test_dir IN LISTS contents)
    if(IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  set_target_properties(${bench_target}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${THRUST_EXECUTABLE_OUTPUT_DIR}"
      CUDA_STANDARD 17
      CXX_STANDARD 17)
  target_link_libraries(${bench_target} PRIVATE nvbench_helper nvbench::main)
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    foreach(thrust_target IN LISTS THRUST_TARGETS)
      thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      string(PREPEND bench_name "${config_prefix}.${bench_prefix}.")
      add_bench(base_bench_target ${bench_name} "${bench_src}")
      target_link_libraries(${bench_name} PRIVATE ${thrust_target})
      thrust_clone_target_properties(${bench_name} ${thrust_target})
    endforeach()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
