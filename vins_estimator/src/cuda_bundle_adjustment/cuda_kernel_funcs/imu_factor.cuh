//
// Created by lmf on 23-7-25.
//

#ifndef CUDA_BA_IMU_FACTOR_CUH
#define CUDA_BA_IMU_FACTOR_CUH

#include "device_utils.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void imu_block_range(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set
);

template<typename T>
__global__ void imu_jacobian_0(     // pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_jacobian_1(     // speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_jacobian_2(     // pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_jacobian_3(     // speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_residual(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_robust_info(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void imu_rhs_0(      // pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void imu_rhs_1(      // speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void imu_rhs_2(      // pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void imu_rhs_3(      // speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_00(     // pose_i, pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_01(     // pose_i, speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_02(     // pose_i, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_03(     // pose_i, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_11(     // speed_bias_i, speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_12(     // speed_bias_i, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_13(     // speed_bias_i, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_22(     // pose_j, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_23(     // pose_j, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void imu_hessian_33(     // speed_bias_j, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_IMU_FACTOR_CUH
