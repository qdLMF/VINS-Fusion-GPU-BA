//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_DEVICE_UTILS_CUH
#define CUDA_BA_DEVICE_UTILS_CUH

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "../common.h"

#include "../device_memory/imu_dev_ptr_set.h"
#include "../device_memory/proj_2f1c_dev_ptr_set.h"
#include "../device_memory/proj_2f2c_dev_ptr_set.h"
#include "../device_memory/proj_1f2c_dev_ptr_set.h"

#include "../cuda_streams/block_range_cuda_stream_set.cuh"
#include "../cuda_streams/proj_temp_cuda_stream_set.cuh"
#include "../cuda_streams/jacobian_residual_cuda_stream_set.cuh"
#include "../cuda_streams/robust_info_cuda_stream_set.cuh"
#include "../cuda_streams/hessian_rhs_cuda_stream_set.cuh"
#include "../cuda_streams/update_cuda_stream_set.cuh"

#define FOCAL_LENGTH 460.0

//#define UNFOLD_READ_1
//#define UNFOLD_READ_2

namespace VINS_FUSION_CUDA_BA {

struct BlockRange{
    unsigned int row_start;
    unsigned int row_end;
    unsigned int col_start;
    unsigned int col_end;
};

template<typename T>
__device__ Eigen::Matrix<T, 3, 3, Eigen::RowMajor> UtilitySkewSymmetric(const Eigen::Matrix<T, 3, 1>& q);

template<typename T>
__device__ Eigen::Quaternion<T> UtilityDeltaQ(const Eigen::Matrix<T, 3, 1> &theta);

template<typename T>
__device__ Eigen::Matrix<T, 4, 4, Eigen::RowMajor> UtilityQLeft(const Eigen::Quaternion<T>& q);

template<typename T>
__device__ Eigen::Matrix<T, 4, 4, Eigen::RowMajor> UtilityQRight(const Eigen::Quaternion<T>& p);

template<typename T>
__device__ void cauchy_loss(T delta, T err2, Eigen::Matrix<T, 3, 1>& rho);

template<typename T>
__device__ void huber_loss(T delta, T err2, Eigen::Matrix<T, 3, 1>& rho);

__device__ BlockRange GetJTJBlockRange(const BlockRange& left, const BlockRange& right);

template<typename T>
__device__ T MyAtomicAdd(T* address, T val);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_DEVICE_UTILS_CUH
