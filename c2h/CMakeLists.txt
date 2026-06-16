cmake_minimum_required(VERSION 3.21)

project(C2H LANGUAGES CXX CUDA)

# when CCCL is used from the CTK, C2H does not need to be rebuilt when making local changes to CCCL (faster iteration)
option(C2H_USE_CCCL_FROM_CTK "Use CCCL from the CTK in c2h." OFF)

# Get dependencies.
cccl_get_catch2()
if (NOT C2H_USE_CCCL_FROM_CTK)
  cccl_get_cccl()
endif()
cccl_get_cudatoolkit()

set(has_curand OFF)
if (TARGET CUDA::curand)
  set(has_curand ON)
endif()

option(C2H_ENABLE_CURAND "Use CUDA CURAND library in c2h." ${has_curand})

# Disable C2H_ENABLE_CURAND if cuRAND is unavailable.
if (C2H_ENABLE_CURAND AND NOT has_curand)
  message(
    WARNING
    "C2H_ENABLE_CURAND is requested, but CUDA::curand is unavailable. Disabling."
  )
  set(C2H_ENABLE_CURAND OFF)
endif()

# Catch2-free core of c2h: shared helpers that must be linkable from targets which do not (and should not) depend on
# Catch2 -- notably the CUB benchmarks. Heavy, reusable instantiations (e.g. the device top-k sort-based reference) live
# here so they are compiled once rather than in every consuming translation unit.
add_library(
  cccl.c2h.core
  STATIC
  device_topk_reference.cu
)
target_include_directories(cccl.c2h.core PUBLIC "${C2H_SOURCE_DIR}/include")
target_link_libraries(cccl.c2h.core PUBLIC cccl.compiler_interface)
if (NOT C2H_USE_CCCL_FROM_CTK)
  target_link_libraries(cccl.c2h.core PUBLIC CCCL::CCCL)
endif()
cccl_configure_target(cccl.c2h.core DIALECT 17)
set_target_properties(
  cccl.c2h.core
  PROPERTIES CXX_VISIBILITY_PRESET "default" CUDA_VISIBILITY_PRESET "default"
)

add_library(
  cccl.c2h
  STATIC
  generators.cu
  generators_gen_values.cu
  generators_uniform_offsets.cu
  generators_vector.cu
)
target_compile_definitions(cccl.c2h PUBLIC CATCH_CONFIG_PREFIX_ALL)
target_link_libraries(
  cccl.c2h
  PUBLIC #
    cccl.c2h.core
    cccl.compiler_interface
    Catch2::Catch2
)
cccl_configure_target(cccl.c2h DIALECT 17)

if (C2H_ENABLE_CURAND)
  target_link_libraries(cccl.c2h PRIVATE CUDA::curand)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=1)
else()
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=0)
endif()

set_target_properties(
  cccl.c2h
  PROPERTIES CXX_VISIBILITY_PRESET "default" CUDA_VISIBILITY_PRESET "default"
)

add_library(cccl.c2h.main OBJECT catch2_runner.cu catch2_runner_helper.cu)
target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
