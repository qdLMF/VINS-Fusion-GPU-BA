//
// Created by lmf on 23-7-22.
//

#include "shape_manager.h"

namespace VINS_FUSION_CUDA_BA {

void ShapeManager::ComputeMaxSizes(
        int max_n_imu_factors,
        int max_n_world_points
) {
    num_key_frames = max_n_imu_factors + 1;
    max_num_imu_factors = max_n_imu_factors;
    max_num_world_points = max_n_world_points;

    jac_pose_ex_num_cols = 6;
    jac_td_num_cols = 1;
    jac_state_num_cols = 15;

    jac_imu_num_rows = 15;
    jac_proj_2f2c_num_rows = 2;
    jac_proj_2f1c_num_rows = 2;
    jac_proj_1f2c_num_rows = 2;

    // ----------

    num_elem_ex_para = 7;
    num_elem_states = num_key_frames * 16;
    num_elem_cur_td = 1;

    num_elem_delta_ex_para = 6;
    num_elem_delta_states = num_key_frames * 15;
    num_elem_delta_cur_td = 1;

    // ----------

    num_rows_Hpp = num_key_frames * jac_state_num_cols + jac_pose_ex_num_cols * 2 + jac_td_num_cols;
    num_cols_Hpp = num_rows_Hpp;

    num_elem_Bpp = num_rows_Hpp;

    num_rows_Hpp_for_marg = num_key_frames * jac_state_num_cols + jac_pose_ex_num_cols * 2 + jac_td_num_cols;
    num_cols_Hpp_for_marg = num_rows_Hpp;

    num_elem_Bpp_for_marg = num_rows_Hpp_for_marg;


    num_rows_H11 = num_rows_Hpp_for_marg - jac_state_num_cols;
    num_cols_H11 = num_cols_Hpp_for_marg - jac_state_num_cols;

    num_rows_H12 = num_rows_H11;
    num_cols_H12 = jac_state_num_cols;

    num_rows_H21 = jac_state_num_cols;
    num_cols_H21 = num_cols_H11;

    num_rows_H22 = jac_state_num_cols;
    num_cols_H22 = jac_state_num_cols;

    num_elem_B11 = num_rows_H11;
    num_elem_B22 = num_rows_H22;

    num_rows_Hprior_eigenvec = num_rows_H11;
    num_cols_Hprior_eigenvec = num_cols_H11;

    num_elem_Hprior_eigenval = num_rows_H11;


    max_num_elem_inv_depth = max_num_world_points;
    max_num_elem_delta_inv_depth = max_num_world_points;
    max_num_elem_Hpm = num_rows_Hpp * max_num_world_points;
    max_num_elem_Hmp = max_num_world_points * num_cols_Hpp;
    max_num_elem_Hmm_diag = max_num_world_points;
    max_num_elem_Bmm = max_num_world_points;
}

void ShapeManager::ComputeActualSizes(
    int n_imu_factors,
    int n_proj_2f1c_factors,
    int n_proj_2f2c_factors,
    int n_proj_1f2c_factors,
    int n_world_points,
    int n_world_points_for_marg
) {
    assert(n_imu_factors <= max_num_imu_factors);
    assert((n_imu_factors + 1) <= num_key_frames);
    assert(n_world_points <= max_num_world_points);
    assert(n_world_points_for_marg <= n_world_points);

    num_imu_factors = n_imu_factors;

    num_proj_2f1c_factors = n_proj_2f1c_factors;
    num_proj_2f2c_factors = n_proj_2f2c_factors;
    num_proj_1f2c_factors = n_proj_1f2c_factors;
    num_world_points = n_world_points;
    num_world_points_for_marg = n_world_points_for_marg;


    jac_inv_depth_num_cols = num_world_points;

    jac_pose_ex_0_col_start = 0;
    jac_pose_ex_0_col_end = jac_pose_ex_0_col_start + jac_pose_ex_num_cols;

    jac_pose_ex_1_col_start = jac_pose_ex_0_col_end;
    jac_pose_ex_1_col_end = jac_pose_ex_1_col_start + jac_pose_ex_num_cols;

    jac_td_col_start = jac_pose_ex_1_col_end;
    jac_td_col_end = jac_td_col_start + jac_td_num_cols;

    jac_state_col_start = jac_td_col_end;
    jac_state_col_end = jac_state_col_start + num_key_frames * jac_state_num_cols;

    jac_inv_depth_col_start = jac_state_col_end;
    jac_inv_depth_col_end = jac_inv_depth_col_start + jac_inv_depth_num_cols;


    jac_imu_row_start = 0;
    jac_imu_row_end = jac_imu_row_start + num_imu_factors * jac_imu_num_rows;

    jac_proj_2f1c_row_start = jac_imu_row_end;
    jac_proj_2f1c_row_end = jac_proj_2f1c_row_start + num_proj_2f1c_factors * jac_proj_2f1c_num_rows;

    jac_proj_2f2c_row_start = jac_proj_2f1c_row_end;
    jac_proj_2f2c_row_end = jac_proj_2f2c_row_start + num_proj_2f2c_factors * jac_proj_2f2c_num_rows;

    jac_proj_1f2c_row_start = jac_proj_2f1c_row_end;
    jac_proj_1f2c_row_end = jac_proj_1f2c_row_start + num_proj_1f2c_factors * jac_proj_1f2c_num_rows;

    // ----------

    num_elem_inv_depth = num_world_points;
    num_elem_delta_inv_depth = num_world_points;

    // ----------

    num_rows_Hmm = num_world_points;
    num_cols_Hmm = num_rows_Hmm;
    num_elem_Hmm_diag = num_rows_Hmm;

    num_rows_Hpm = num_rows_Hpp;
    num_cols_Hpm = num_cols_Hmm;

    num_rows_Hmp = num_rows_Hmm;
    num_cols_Hmp = num_cols_Hpp;

    num_elem_Bmm = num_rows_Hmm;


    num_rows_Hmm_for_marg = num_world_points_for_marg;
    num_cols_Hmm_for_marg = num_rows_Hmm_for_marg;
    num_elem_Hmm_diag_for_marg = num_rows_Hmm_for_marg;

    num_rows_Hpm_for_marg = num_rows_Hpp_for_marg;
    num_cols_Hpm_for_marg = num_cols_Hmm_for_marg;

    num_rows_Hmp_for_marg = num_rows_Hmm_for_marg;
    num_cols_Hmp_for_marg = num_cols_Hpp_for_marg;

    num_elem_Bmm_for_marg = num_rows_Hmm_for_marg;
}

} // namespace VINS_FUSION_CUDA_BA

