# Fetch nvbench
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_library(nvbench_helper OBJECT nvbench_helper/nvbench_helper.cuh 
                                  nvbench_helper/nvbench_helper.cu)

target_link_libraries(nvbench_helper PUBLIC CUB::CUB 
                                            Thrust::Thrust 
                                            CUB::libcudacxx 
                                            nvbench::nvbench
                                     PRIVATE CUDA::curand)

target_include_directories(nvbench_helper PUBLIC "${CMAKE_CURRENT_LIST_DIR}/nvbench_helper")
set_target_properties(nvbench_helper PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)

CPMAddPackage("gh:catchorg/Catch2@2.13.9")
add_executable(nvbench_helper.test test/gen_entropy.cu
                                   test/gen_distribution.cu
                                   test/main.cpp)
target_link_libraries(nvbench_helper.test PRIVATE nvbench_helper Catch2::Catch2)
set_target_properties(nvbench_helper.test
       PROPERTIES
              ARCHIVE_OUTPUT_DIRECTORY "${CUB_LIBRARY_OUTPUT_DIR}"
              LIBRARY_OUTPUT_DIRECTORY "${CUB_LIBRARY_OUTPUT_DIR}"
              RUNTIME_OUTPUT_DIRECTORY "${CUB_EXECUTABLE_OUTPUT_DIR}"
              CUDA_STANDARD 17
              CXX_STANDARD 17)
