//
// Created by lmf on 23-7-22.
//

#include "proj_1f2c_dev_ptr_set.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void Proj1F2CFactorDevPtrSet<T>::SetNullptr() {
    // input
    pts_i = nullptr;
    pts_j = nullptr;
    velocity_i = nullptr;
    velocity_j = nullptr;
    td_i = nullptr;
    td_j = nullptr;
    p_ex_0 = nullptr;
    q_ex_0 = nullptr;
    p_ex_1 = nullptr;
    q_ex_1 = nullptr;
    inv_depth = nullptr;
    cur_td = nullptr;
    idx_i = nullptr;
    idx_j = nullptr;
    inv_depth_idx = nullptr;
    inv_depth_idx_for_marg = nullptr;
    involved_in_marg = nullptr;

    pose_ex_0_is_fixed = nullptr;
    pose_ex_1_is_fixed = nullptr;
    inv_depth_is_fixed = nullptr;
    cur_td_is_fixed = nullptr;

    // temp
    pts_i_td = nullptr;
    pts_j_td = nullptr;
    pts_cam_i = nullptr;
    pts_imu_i = nullptr;
    pts_imu_j = nullptr;
    pts_cam_j = nullptr;
    reduce = nullptr;
    r_ex_0 = nullptr;
    r_ex_0_trans = nullptr;
    r_ex_1 = nullptr;
    r_ex_1_trans = nullptr;
    tmp_r_1 = nullptr;

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
