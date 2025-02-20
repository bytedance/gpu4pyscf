# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(gvhf_mixed_precision SHARED
  rys_contract_jk.cu rys_jk_driver.cu rys_roots_dat.cu rys_roots_dat_single_precision.cu
  unrolled_os.cu unrolled_rys.cu
)

set_target_properties(gvhf_mixed_precision PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf_mixed_precision OpenMP::OpenMP_C)
