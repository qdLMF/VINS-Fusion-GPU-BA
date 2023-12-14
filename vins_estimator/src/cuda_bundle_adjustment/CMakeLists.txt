cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/bin/nvcc)
set(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w")	# "-w" means "ignore all warnings"

find_package(CUDA REQUIRED)
message(STATUS "CUDA_LIBRARIES : ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

add_library(CUDAVinsFusionBundleAdjustment SHARED
	cuda_kernel_funcs/cublas_funcs.cuh
	cuda_kernel_funcs/cublas_funcs.cu
	cuda_kernel_funcs/cusolver_funcs.cuh
	cuda_kernel_funcs/cusolver_funcs.cu
	cuda_kernel_funcs/device_utils.cuh
	cuda_kernel_funcs/device_utils.cu
	cuda_kernel_funcs/imu_factor.cuh
	cuda_kernel_funcs/imu_factor.cu
	cuda_kernel_funcs/proj_1f2c_factor.cuh
	cuda_kernel_funcs/proj_1f2c_factor.cu
	cuda_kernel_funcs/proj_2f1c_factor.cuh
	cuda_kernel_funcs/proj_2f1c_factor.cu
	cuda_kernel_funcs/proj_2f2c_factor.cuh
	cuda_kernel_funcs/proj_2f2c_factor.cu
	cuda_kernel_funcs/launch_kernels.cuh
	cuda_kernel_funcs/launch_kernels.cu
	cuda_kernel_funcs/solve_delta.cuh
	cuda_kernel_funcs/solve_delta.cu
	cuda_kernel_funcs/update_states.cuh
	cuda_kernel_funcs/update_states.cu
	cuda_streams/block_range_cuda_stream_set.cuh
	cuda_streams/block_range_cuda_stream_set.cu
	cuda_streams/hessian_rhs_cuda_stream_set.cuh
	cuda_streams/hessian_rhs_cuda_stream_set.cu
	cuda_streams/jacobian_residual_cuda_stream_set.cuh
	cuda_streams/jacobian_residual_cuda_stream_set.cu
	cuda_streams/proj_temp_cuda_stream_set.cuh
	cuda_streams/proj_temp_cuda_stream_set.cu
	cuda_streams/robust_info_cuda_stream_set.cuh
	cuda_streams/robust_info_cuda_stream_set.cu
	cuda_streams/update_cuda_stream_set.cuh
	cuda_streams/update_cuda_stream_set.cu
	#
	device_memory/GPUMatrix.h
	device_memory/GPUMatrix_impl.hpp
	device_memory/cuda_error_check.h
	device_memory/imu_gpu_mem_pool.h
	device_memory/imu_dev_ptr_set.h
	device_memory/imu_dev_ptr_set.cpp
	device_memory/proj_1f2c_gpu_mem_pool.h
	device_memory/proj_1f2c_dev_ptr_set.h
	device_memory/proj_1f2c_dev_ptr_set.cpp
	device_memory/proj_2f1c_gpu_mem_pool.h
	device_memory/proj_2f1c_dev_ptr_set.h
	device_memory/proj_2f1c_dev_ptr_set.cpp
	device_memory/proj_2f2c_gpu_mem_pool.h
	device_memory/proj_2f2c_dev_ptr_set.h
	device_memory/proj_2f2c_dev_ptr_set.cpp
	#
	common.h
	factors.h
	factors.cpp
	imu_allocator.h
	imu_allocator.cpp
	proj_allocators.h
	proj_allocators.cpp
	levenberg_lambda.h
	levenberg_lambda.cpp
	shape_manager.h	
	shape_manager.cpp
	sliding_window.h
	sliding_window.cpp
	utility.h
	utility.cpp
	simple_marg_factor.h
	simple_marg_factor.cpp
)
target_link_libraries(CUDAVinsFusionBundleAdjustment ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
set_target_properties(CUDAVinsFusionBundleAdjustment PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


