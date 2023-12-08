//
// Created by lmf on 23-7-26.
//

#ifndef CUDA_BA_LAUNCH_KERNELS_CUH
#define CUDA_BA_LAUNCH_KERNELS_CUH

#include "device_utils.cuh"

#include "imu_factor.cuh"
#include "proj_2f1c_factor.cuh"
#include "proj_2f2c_factor.cuh"
#include "proj_1f2c_factor.cuh"

#include "update_states.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void AllBlockRangeKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool use_imu,
    bool is_stereo
);

template<typename T>
void AllProjTempKernels(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void AllJacobianAndResidualKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);

template<typename T>
void AllRobustInfoKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);

template<typename T>
void AllHessianAndRHSKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    T* Hpp,
    int leading_dim_Hpp,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    T* Hmm_diag,
    T* Bpp,
    T* Bmm,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);

template<typename T>
void AllUpdateKernels(
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    UpdateStreamSet& update_cuda_stream
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void AllProjBlockRangeKernelsForMarg(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool is_stereo
);

template<typename T>
void AllProjTempKernelsForMarg(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);

template<typename T>
void AllJacobianAndResidualKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo
);

template<typename T>
void AllRobustInfoKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);

template<typename T>
void AllHessianAndRHSKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    T* Hpp,
    int leading_dim_Hpp,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    T* Hmm_diag,
    T* Bpp,
    T* Bmm,
    bool use_imu,
    bool is_stereo
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_LAUNCH_KERNELS_CUH