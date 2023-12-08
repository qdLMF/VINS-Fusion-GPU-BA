//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_PROJ_2F1C_DEV_PTR_SET_H
#define CUDA_BA_PROJ_2F1C_DEV_PTR_SET_H

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class Proj2F1CFactorDevPtrSet {
public :
    // input
    T *pts_i = nullptr;                     // inner dim * outter dim : num_projection_factors * 3
    T *pts_j = nullptr;                     // inner dim * outter dim : num_projection_factors * 3
    T *velocity_i = nullptr;                // inner dim * outter dim : num_projection_factors * 3
    T *velocity_j = nullptr;                // inner dim * outter dim : num_projection_factors * 3
    T *td_i = nullptr;                      // inner dim * outter dim : num_projection_factors * 1
    T *td_j = nullptr;                      // inner dim * outter dim : num_projection_factors * 1
    T *p_i = nullptr;                       // inner dim * outter dim : num_projection_factors * 3
    T *q_i = nullptr;                       // inner dim * outter dim : num_projection_factors * 4
    T *p_j = nullptr;                       // inner dim * outter dim : num_projection_factors * 3
    T *q_j = nullptr;                       // inner dim * outter dim : num_projection_factors * 4
    T *p_ex_0 = nullptr;                    // inner dim * outter dim : num_projection_factors * 3
    T *q_ex_0 = nullptr;                    // inner dim * outter dim : num_projection_factors * 4
    T *inv_depth = nullptr;                 // inner dim * outter dim : num_projection_factors * 1
    T *cur_td = nullptr;                    // inner dim * outter dim : num_projection_factors * 1
    int *idx_i = nullptr;                   // inner dim * outter dim : num_projection_factors * 1
    int *idx_j = nullptr;                   // inner dim * outter dim : num_projection_factors * 1
    int *inv_depth_idx = nullptr;           // inner dim * outter dim : num_projection_factors * 1
    int *inv_depth_idx_for_marg = nullptr;  // inner dim * outter dim : num_projection_factors * 1
    char *involved_in_marg = nullptr;       // inner dim * outter dim : num_projection_factors * 1

    char* pose_i_is_fixed = nullptr;        // inner dim * outter dim : num_projection_factors * 1
    char* pose_j_is_fixed = nullptr;        // inner dim * outter dim : num_projection_factors * 1
    char* pose_ex_0_is_fixed = nullptr;     // inner dim * outter dim : num_projection_factors * 1
    char* inv_depth_is_fixed = nullptr;     // inner dim * outter dim : num_projection_factors * 1
    char* cur_td_is_fixed = nullptr;        // inner dim * outter dim : num_projection_factors * 1

    // temp
    T *pts_i_td = nullptr;   // num_proj_factors * 3
    T *pts_j_td = nullptr;   // num_proj_factors * 3
    T *pts_cam_i = nullptr;  // num_proj_factors * 3
    T *pts_imu_i = nullptr;  // num_proj_factors * 3
    T *pts_world = nullptr;  // num_proj_factors * 3
    T *pts_imu_j = nullptr;  // num_proj_factors * 3
    T *pts_cam_j = nullptr;  // num_proj_factors * 3
    T *reduce = nullptr;     // num_proj_factors * 6 = 2 x 3
    T *r_i = nullptr;        // num_proj_factors * 9 = 3 x 3
    T *r_i_trans = nullptr;  // num_proj_factors * 9 = 3 x 3
    T *r_j = nullptr;        // num_proj_factors * 9 = 3 x 3
    T *r_j_trans = nullptr;  // num_proj_factors * 9 = 3 x 3
    T *r_ex = nullptr;       // num_proj_factors * 9 = 3 x 3
    T *r_ex_trans = nullptr; // num_proj_factors * 9 = 3 x 3
    T *tmp_r_1 = nullptr;    // num_proj_factors * 9 = 3 x 3
    T *tmp_r_2 = nullptr;    // num_proj_factors * 9 = 3 x 3

    // output
    T *robust_info = nullptr;          // num_projection_factors * 4 = 2 x 2
    T *robust_chi2 = nullptr;          // num_projection_factors * 1
    // T *drho = nullptr;                 // num_projection_factors * 1
    T *residual = nullptr;             // num_projection_factors * 2
    T *drho_info_residual = nullptr;   // num_projection_factors * 2
    T *jacobian_0 = nullptr;           // num_projection_factors * 12 = 2 x 6 , wrt p_i, q_i
    T *jacobian_1 = nullptr;           // num_projection_factors * 12 = 2 x 6 , wrt p_j, q_j
    T *jacobian_2 = nullptr;           // num_projection_factors * 12 = 2 x 6 , wrt p_ex_0, q_ex_0
    T *jacobian_3 = nullptr;           // num_projection_factors *  2 = 2 x 1 , wrt inv_depth
    T *jacobian_4 = nullptr;           // num_projection_factors *  2 = 2 x 1 , wrt cur_td

    // T *hessian_00 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_01 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_02 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_03 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpm
    // T *hessian_04 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpp
    // T *hessian_10 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_11 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_12 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_13 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpm
    // T *hessian_14 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpp
    // T *hessian_20 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_21 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_22 = nullptr;           // num_projection_factors * 36 = 6 x 6 , in Hpp
    // T *hessian_23 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpm
    // T *hessian_24 = nullptr;           // num_projection_factors *  6 = 6 x 1 , in Hpp
    // T *hessian_30 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hmp
    // T *hessian_31 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hmp
    // T *hessian_32 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hmp
    // T *hessian_33 = nullptr;           // num_projection_factors *  1 = 1 x 1 , in Hmm
    // T *hessian_34 = nullptr;           // num_projection_factors *  1 = 1 x 1 , in Hmp
    // T *hessian_40 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hpp
    // T *hessian_41 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hpp
    // T *hessian_42 = nullptr;           // num_projection_factors *  6 = 1 x 6 , in Hpp
    // T *hessian_43 = nullptr;           // num_projection_factors *  1 = 1 x 1 , in Hpm
    // T *hessian_44 = nullptr;           // num_projection_factors *  1 = 1 x 1 , in Hpp

    // T *rhs_0 = nullptr;                // num_projection_factors *  6 = 6 x 1 , in Bpp
    // T *rhs_1 = nullptr;                // num_projection_factors *  6 = 6 x 1 , in Bpp
    // T *rhs_2 = nullptr;                // num_projection_factors *  6 = 6 x 1 , in Bpp
    // T *rhs_3 = nullptr;                // num_projection_factors *  1 = 1 x 1 , in Bmm
    // T *rhs_4 = nullptr;                // num_projection_factors *  1 = 1 x 1 , in Bpp

    unsigned int *hessian_00_row_start = nullptr;
    unsigned int *hessian_00_col_start = nullptr;
    unsigned int *hessian_01_row_start = nullptr;
    unsigned int *hessian_01_col_start = nullptr;
    unsigned int *hessian_02_row_start = nullptr;
    unsigned int *hessian_02_col_start = nullptr;
    unsigned int *hessian_03_row_start = nullptr;
    unsigned int *hessian_03_col_start = nullptr;
    unsigned int *hessian_04_row_start = nullptr;
    unsigned int *hessian_04_col_start = nullptr;

    unsigned int *hessian_10_row_start = nullptr;
    unsigned int *hessian_10_col_start = nullptr;
    unsigned int *hessian_11_row_start = nullptr;
    unsigned int *hessian_11_col_start = nullptr;
    unsigned int *hessian_12_row_start = nullptr;
    unsigned int *hessian_12_col_start = nullptr;
    unsigned int *hessian_13_row_start = nullptr;
    unsigned int *hessian_13_col_start = nullptr;
    unsigned int *hessian_14_row_start = nullptr;
    unsigned int *hessian_14_col_start = nullptr;

    unsigned int *hessian_20_row_start = nullptr;
    unsigned int *hessian_20_col_start = nullptr;
    unsigned int *hessian_21_row_start = nullptr;
    unsigned int *hessian_21_col_start = nullptr;
    unsigned int *hessian_22_row_start = nullptr;
    unsigned int *hessian_22_col_start = nullptr;
    unsigned int *hessian_23_row_start = nullptr;
    unsigned int *hessian_23_col_start = nullptr;
    unsigned int *hessian_24_row_start = nullptr;
    unsigned int *hessian_24_col_start = nullptr;

    unsigned int *hessian_30_row_start = nullptr;
    unsigned int *hessian_30_col_start = nullptr;
    unsigned int *hessian_31_row_start = nullptr;
    unsigned int *hessian_31_col_start = nullptr;
    unsigned int *hessian_32_row_start = nullptr;
    unsigned int *hessian_32_col_start = nullptr;
    unsigned int *hessian_33_row_start = nullptr;
    unsigned int *hessian_33_col_start = nullptr;
    unsigned int *hessian_34_row_start = nullptr;
    unsigned int *hessian_34_col_start = nullptr;

    unsigned int *hessian_40_row_start = nullptr;
    unsigned int *hessian_40_col_start = nullptr;
    unsigned int *hessian_41_row_start = nullptr;
    unsigned int *hessian_41_col_start = nullptr;
    unsigned int *hessian_42_row_start = nullptr;
    unsigned int *hessian_42_col_start = nullptr;
    unsigned int *hessian_43_row_start = nullptr;
    unsigned int *hessian_43_col_start = nullptr;
    unsigned int *hessian_44_row_start = nullptr;
    unsigned int *hessian_44_col_start = nullptr;

    unsigned int *rhs_0_row_start = nullptr;
    unsigned int *rhs_1_row_start = nullptr;
    unsigned int *rhs_2_row_start = nullptr;
    unsigned int *rhs_3_row_start = nullptr;
    unsigned int *rhs_4_row_start = nullptr;

public :
    void SetNullptr();
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_PROJ_2F1C_DEV_PTR_SET_H
