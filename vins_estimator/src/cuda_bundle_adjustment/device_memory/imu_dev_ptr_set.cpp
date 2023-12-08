//
// Created by lmf on 23-7-22.
//

#include "imu_dev_ptr_set.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void IMUFactorDevPtrSet<T>::SetNullptr() {
    // input
    gravity = nullptr;
    p_i = nullptr;
    q_i = nullptr;
    v_i = nullptr;
    bias_acc_i = nullptr;
    bias_gyr_i = nullptr;
    p_j = nullptr;
    q_j = nullptr;
    v_j = nullptr;
    bias_acc_j = nullptr;
    bias_gyr_j = nullptr;
    sum_dt = nullptr;
    linearized_bias_acc = nullptr;
    linearized_bias_gyr = nullptr;
    delta_p = nullptr;
    delta_q = nullptr;
    delta_v = nullptr;
    jacobian_p_bias_acc = nullptr;
    jacobian_p_bias_gyr = nullptr;
    jacobian_q_bias_gyr = nullptr;
    jacobian_v_bias_acc = nullptr;
    jacobian_v_bias_gyr = nullptr;
    covariance = nullptr;
    info = nullptr;
    sqrt_info = nullptr;
    idx_i = nullptr;
    idx_j = nullptr;
    involved_in_marg = nullptr;

    pose_i_is_fixed = nullptr;
    speed_bias_i_is_fixed = nullptr;
    pose_j_is_fixed = nullptr;
    speed_bias_j_is_fixed = nullptr;

    // output
    robust_info = nullptr;
    robust_chi2 = nullptr;
    // drho = nullptr;
    residual = nullptr;
    drho_info_residual = nullptr;
    jacobian_0 = nullptr;
    jacobian_1 = nullptr;
    jacobian_2 = nullptr;
    jacobian_3 = nullptr;
    // hessian_00 = nullptr;
    // hessian_01 = nullptr;
    // hessian_02 = nullptr;
    // hessian_03 = nullptr;
    // hessian_10 = nullptr;
    // hessian_11 = nullptr;
    // hessian_12 = nullptr;
    // hessian_13 = nullptr;
    // hessian_20 = nullptr;
    // hessian_21 = nullptr;
    // hessian_22 = nullptr;
    // hessian_23 = nullptr;
    // hessian_30 = nullptr;
    // hessian_31 = nullptr;
    // hessian_32 = nullptr;
    // hessian_33 = nullptr;
    // rhs_0 = nullptr;
    // rhs_1 = nullptr;
    // rhs_2 = nullptr;
    // rhs_3 = nullptr;

    hessian_00_row_start = nullptr; hessian_00_col_start = nullptr;
    hessian_01_row_start = nullptr; hessian_01_col_start = nullptr;
    hessian_02_row_start = nullptr; hessian_02_col_start = nullptr;
    hessian_03_row_start = nullptr; hessian_03_col_start = nullptr;

    hessian_10_row_start = nullptr; hessian_10_col_start = nullptr;
    hessian_11_row_start = nullptr; hessian_11_col_start = nullptr;
    hessian_12_row_start = nullptr; hessian_12_col_start = nullptr;
    hessian_13_row_start = nullptr; hessian_13_col_start = nullptr;

    hessian_20_row_start = nullptr; hessian_20_col_start = nullptr;
    hessian_21_row_start = nullptr; hessian_21_col_start = nullptr;
    hessian_22_row_start = nullptr; hessian_22_col_start = nullptr;
    hessian_23_row_start = nullptr; hessian_23_col_start = nullptr;

    hessian_30_row_start = nullptr; hessian_30_col_start = nullptr;
    hessian_31_row_start = nullptr; hessian_31_col_start = nullptr;
    hessian_32_row_start = nullptr; hessian_32_col_start = nullptr;
    hessian_33_row_start = nullptr; hessian_33_col_start = nullptr;

    rhs_0_row_start = nullptr;
    rhs_1_row_start = nullptr;
    rhs_2_row_start = nullptr;
    rhs_3_row_start = nullptr;
}

} // namespace VINS_FUSION_CUDA_BA

