function(cccl_c_parallel_add_example source)
  get_filename_component(target_name "${source}" NAME_WE)
  set(target_name "cccl.c.parallel.example.${target_name}")

  cccl_add_executable(
    ${target_name}
    NO_METATARGETS
    DIALECT 20
    SOURCES "${source}"
  )

  set_target_properties(${target_name} PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
  target_link_libraries(
    ${target_name}
    PRIVATE
      cccl.compiler_interface
      cccl.c.parallel
      CUDA::cudart_static
      CUDA::cuda_driver
  )

  list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_FIRST_INCLUDE_DIR)
  target_compile_definitions(
    ${target_name}
    PRIVATE
      CCCL_EXAMPLE_CUB_PATH="-I${CCCL_SOURCE_DIR}/cub"
      CCCL_EXAMPLE_THRUST_PATH="-I${CCCL_SOURCE_DIR}/thrust"
      CCCL_EXAMPLE_LIBCUDACXX_PATH="-I${CCCL_SOURCE_DIR}/libcudacxx/include"
      CCCL_EXAMPLE_CTK_PATH="-I${CUDA_FIRST_INCLUDE_DIR}"
  )
endfunction()

file(
  GLOB example_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cpp
  *.cu
)

foreach (example_src IN LISTS example_srcs)
  cccl_c_parallel_add_example("${example_src}")
endforeach()
