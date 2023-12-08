//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_PROJ_2F2C_FACTOR_CUH
#define CUDA_BA_PROJ_2F2C_FACTOR_CUH

#include "device_utils.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void proj_2f2c_block_range(
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_temp(
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_0_l(     // p_i
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_0_r(     // q_i
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_1_l(     // p_j
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_1_r(     // q_j
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_2_l(     // p_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_2_r(     // q_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_3_l(     // p_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_3_r(     // q_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_4(   // inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_jacobian_5(   // cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_residual(
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_robust_info(
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_0(    // p_i, q_i
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_1(    // p_j, q_j
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_2(    // p_ex_0, q_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_3(    // p_ex_1, q_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_4(    // inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bmm,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_rhs_5(    // cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_00(   // pose_i, pose_i
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_01(   // pose_i, pose_j
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_02(   // pose_i, pose_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_03(   // pose_i, pose_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_04(   // pose_i, inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_05(   // pose_i, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_11(   // pose_j, pose_j
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_12(   // pose_j, pose_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_13(   // pose_j, pose_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_14(   // pose_j, inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_15(   // pose_j, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_22(   // pose_ex_0, pose_ex_0
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_23(   // pose_ex_0, pose_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_24(   // pose_ex_0, inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_25(   // pose_ex_0, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_33(   // pose_ex_1, pose_ex_1
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_34(   // pose_ex_1, inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_35(   // pose_ex_1, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_44(   // inv_depth, inv_depth
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hmm_diag,
    bool marg
);

template<typename T>
__global__ void proj_2f2c_hessian_45(   // inv_depth, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template<typename T>__global__ void proj_2f2c_hessian_55(   // cur_td, cur_td
    int num_proj_factors,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_PROJ_2F2C_FACTOR_CUH
