//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_SHAPE_MANAGER_H
#define CUDA_BA_SHAPE_MANAGER_H

#include <vector>

#include "factors.h"

namespace VINS_FUSION_CUDA_BA {

struct ShapeManager {
public :
    int num_key_frames = 0;
    int max_num_imu_factors = 0;

    int num_imu_factors = 0;

    int num_proj_2f2c_factors = 0;
    int num_proj_2f1c_factors = 0;
    int num_proj_1f2c_factors = 0;
    int num_world_points = 0;
    int num_world_points_for_marg = 0;

public :
    int jac_pose_ex_num_cols = 0;
    int jac_td_num_cols = 0;
    int jac_state_num_cols =  0;
    int jac_inv_depth_num_cols = 0;

    int jac_pose_ex_0_col_start = 0;
    int jac_pose_ex_0_col_end = 0;

    int jac_pose_ex_1_col_start = 0;
    int jac_pose_ex_1_col_end = 0;

    int jac_td_col_start = 0;
    int jac_td_col_end = 0;

    int jac_state_col_start = 0;
    int jac_state_col_end = 0;

    int jac_inv_depth_col_start = 0;
    int jac_inv_depth_col_end = 0;

    int jac_imu_num_rows = 0;
    int jac_proj_2f2c_num_rows = 0;
    int jac_proj_2f1c_num_rows = 0;
    int jac_proj_1f2c_num_rows = 0;

    int jac_imu_row_start = 0;
    int jac_imu_row_end = 0;

    int jac_proj_2f2c_row_start = 0;
    int jac_proj_2f2c_row_end = 0;

    int jac_proj_2f1c_row_start = 0;
    int jac_proj_2f1c_row_end = 0;

    int jac_proj_1f2c_row_start = 0;
    int jac_proj_1f2c_row_end = 0;

public :
    int num_elem_ex_para = 0;
    int num_elem_states = 0;
    int num_elem_inv_depth = 0;
    int num_elem_cur_td = 0;

    int num_elem_delta_ex_para = 0;
    int num_elem_delta_states = 0;
    int num_elem_delta_inv_depth = 0;
    int num_elem_delta_cur_td = 0;

public :
    int num_rows_Hpp;
    int num_cols_Hpp;

    int num_rows_Hpm;
    int num_cols_Hpm;

    int num_rows_Hmp;
    int num_cols_Hmp;

    int num_rows_Hmm;
    int num_cols_Hmm;
    int num_elem_Hmm_diag;

    int num_elem_Bpp;
    int num_elem_Bmm;

    int num_rows_Hpp_for_marg;
    int num_cols_Hpp_for_marg;

    int num_rows_Hpm_for_marg;
    int num_cols_Hpm_for_marg;

    int num_rows_Hmp_for_marg;
    int num_cols_Hmp_for_marg;

    int num_rows_Hmm_for_marg;
    int num_cols_Hmm_for_marg;
    int num_elem_Hmm_diag_for_marg;

    int num_elem_Bpp_for_marg;
    int num_elem_Bmm_for_marg;

    int num_rows_H11;
    int num_cols_H11;

    int num_rows_H12;
    int num_cols_H12;

    int num_rows_H21;
    int num_cols_H21;

    int num_rows_H22;
    int num_cols_H22;

    int num_elem_B11;
    int num_elem_B22;

    int num_rows_Hprior_eigenvec;
    int num_cols_Hprior_eigenvec;

    int num_elem_Hprior_eigenval;

public :
    int max_num_world_points = 0;
    int max_num_elem_inv_depth = 0;
    int max_num_elem_delta_inv_depth = 0;
    int max_num_elem_Hpm = 0;
    int max_num_elem_Hmp = 0;
    int max_num_elem_Hmm_diag = 0;
    int max_num_elem_Bmm = 0;

public :
    void ComputeMaxSizes(
        int n_imu_factors,
        int max_n_world_points
    );

    void ComputeActualSizes(
        int n_imu_factors,
        int n_proj_2f1c_factors,
        int n_proj_2f2c_factors,
        int n_proj_1f2c_factors,
        int n_world_points,
        int n_world_points_for_marg
    );
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_SHAPE_MANAGER_H
