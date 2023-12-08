//
// Created by lmf on 23-7-23.
//

#include "proj_1f2c_factor.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void proj_1f2c_block_range(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    int idx_inv_depth;
    if(!marg) {
        idx_inv_depth = *(dev_ptr_set.inv_depth_idx + idx_factor);
    } else {
        idx_inv_depth = *(dev_ptr_set.inv_depth_idx_for_marg + idx_factor);
        if( idx_inv_depth == -1 ) {
            return;
        }
    }

    // pose_ex_0
    unsigned int jacobian_0_col_start = 0;
    unsigned int jacobian_0_col_end = jacobian_0_col_start + 6;

    // pose_ex_1
    unsigned int jacobian_1_col_start = 6;
    unsigned int jacobian_1_col_end = jacobian_1_col_start + 6;

    // inv_depth
    unsigned int jacobian_2_col_start = idx_inv_depth;
    unsigned int jacobian_2_col_end = jacobian_2_col_start + 1;

    // cur_td
    unsigned int jacobian_3_col_start = 12;
    unsigned int jacobian_3_col_end = jacobian_3_col_start + 1;

    BlockRange jacobian_0_block_range{0, 1, jacobian_0_col_start, jacobian_0_col_end};
    BlockRange jacobian_1_block_range{0, 1, jacobian_1_col_start, jacobian_1_col_end};
    BlockRange jacobian_2_block_range{0, 1, jacobian_2_col_start, jacobian_2_col_end};
    BlockRange jacobian_3_block_range{0, 1, jacobian_3_col_start, jacobian_3_col_end};

    BlockRange hessian_00_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_00_row_start + idx_factor) = hessian_00_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_00_col_start + idx_factor) = hessian_00_block_range.col_start;    // in Hpp
    BlockRange hessian_01_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_01_row_start + idx_factor) = hessian_01_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_01_col_start + idx_factor) = hessian_01_block_range.col_start;    // in Hpp
    BlockRange hessian_02_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_02_row_start + idx_factor) = hessian_02_block_range.row_start;    // in Hpm
    *(dev_ptr_set.hessian_02_col_start + idx_factor) = hessian_02_block_range.col_start;    // in Hpm
    BlockRange hessian_03_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_03_row_start + idx_factor) = hessian_03_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_03_col_start + idx_factor) = hessian_03_block_range.col_start;    // in Hpp

    BlockRange hessian_10_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_10_row_start + idx_factor) = hessian_10_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_10_col_start + idx_factor) = hessian_10_block_range.col_start;    // in Hpp
    BlockRange hessian_11_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_11_row_start + idx_factor) = hessian_11_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_11_col_start + idx_factor) = hessian_11_block_range.col_start;    // in Hpp
    BlockRange hessian_12_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_12_row_start + idx_factor) = hessian_12_block_range.row_start;    // in Hpm
    *(dev_ptr_set.hessian_12_col_start + idx_factor) = hessian_12_block_range.col_start;    // in Hpm
    BlockRange hessian_13_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_13_row_start + idx_factor) = hessian_13_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_13_col_start + idx_factor) = hessian_13_block_range.col_start;    // in Hpp

    BlockRange hessian_20_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_20_row_start + idx_factor) = hessian_20_block_range.row_start;    // in Hmp
    *(dev_ptr_set.hessian_20_col_start + idx_factor) = hessian_20_block_range.col_start;    // in Hmp
    BlockRange hessian_21_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_21_row_start + idx_factor) = hessian_21_block_range.row_start;    // in Hmp
    *(dev_ptr_set.hessian_21_col_start + idx_factor) = hessian_21_block_range.col_start;    // in Hmp
    BlockRange hessian_22_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_22_row_start + idx_factor) = hessian_22_block_range.row_start;    // in Hmm
    *(dev_ptr_set.hessian_22_col_start + idx_factor) = hessian_22_block_range.col_start;    // in Hmm
    BlockRange hessian_23_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_23_row_start + idx_factor) = hessian_23_block_range.row_start;    // in Hmp
    *(dev_ptr_set.hessian_23_col_start + idx_factor) = hessian_23_block_range.col_start;    // in Hmp

    BlockRange hessian_30_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_30_row_start + idx_factor) = hessian_30_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_30_col_start + idx_factor) = hessian_30_block_range.col_start;    // in Hpp
    BlockRange hessian_31_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_31_row_start + idx_factor) = hessian_31_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_31_col_start + idx_factor) = hessian_31_block_range.col_start;    // in Hpp
    BlockRange hessian_32_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_32_row_start + idx_factor) = hessian_32_block_range.row_start;    // in Hpm
    *(dev_ptr_set.hessian_32_col_start + idx_factor) = hessian_32_block_range.col_start;    // in Hpm
    BlockRange hessian_33_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_33_row_start + idx_factor) = hessian_33_block_range.row_start;    // in Hpp
    *(dev_ptr_set.hessian_33_col_start + idx_factor) = hessian_33_block_range.col_start;    // in Hpp

    BlockRange rhs_0_block_range{jacobian_0_col_start, jacobian_0_col_end, 0, 1};
    *(dev_ptr_set.rhs_0_row_start + idx_factor) = rhs_0_block_range.row_start;  // in Bpp
    BlockRange rhs_1_block_range{jacobian_1_col_start, jacobian_1_col_end, 0, 1};
    *(dev_ptr_set.rhs_1_row_start + idx_factor) = rhs_1_block_range.row_start;  // in Bpp
    BlockRange rhs_2_block_range{jacobian_2_col_start, jacobian_2_col_end, 0, 1};
    *(dev_ptr_set.rhs_2_row_start + idx_factor) = rhs_2_block_range.row_start;  // in Bmm
    BlockRange rhs_3_block_range{jacobian_3_col_start, jacobian_3_col_end, 0, 1};
    *(dev_ptr_set.rhs_3_row_start + idx_factor) = rhs_3_block_range.row_start;  // in Bpp
}

template<typename T>
__global__ void proj_1f2c_temp(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T cur_td; cur_td = *(dev_ptr_set.cur_td + idx_factor);

    T td_i; td_i = *(dev_ptr_set.td_i + idx_factor);
    T td_j; td_j = *(dev_ptr_set.td_j + idx_factor);

    Eigen::Matrix<T, 3, 1> pts_i;
    for(int i = 0; i < pts_i.size(); i++) {
        pts_i(i) = *(dev_ptr_set.pts_i + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> velocity_i;
    for(int i = 0; i < velocity_i.size(); i++) {
        velocity_i(i) = *(dev_ptr_set.velocity_i + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> pts_j;
    for(int i = 0; i < pts_j.size(); i++) {
        pts_j(i) = *(dev_ptr_set.pts_j + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> velocity_j;
    for(int i = 0; i < velocity_j.size(); i++) {
        velocity_j(i) = *(dev_ptr_set.velocity_j + i * num_proj_factors + idx_factor);
    }

    T inv_depth_i = *(dev_ptr_set.inv_depth + idx_factor);

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Quaternion<T> q;
    Eigen::Matrix<T, 3, 1> p;
    Eigen::Matrix<T, 3, 1> pts;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_trans;

    Eigen::Matrix<T, 3, 1> pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    for(int i = 0; i < pts_i_td.size(); i++) {
        *(dev_ptr_set.pts_i_td + i * num_proj_factors + idx_factor) = pts_i_td(i);
    }

    Eigen::Matrix<T, 3, 1> pts_j_td = pts_j - (cur_td - td_j) * velocity_j;
    for(int i = 0; i < pts_j_td.size(); i++) {
        *(dev_ptr_set.pts_j_td + i * num_proj_factors + idx_factor) = pts_j_td(i);
    }

    pts = pts_i_td / inv_depth_i;    // pts_cam_i = pts_i_td / inv_depth_i;
    for(int i = 0; i < pts.size(); i++) {
        *(dev_ptr_set.pts_cam_i + i * num_proj_factors + idx_factor) = pts(i);
    }

    // q_ex_0, p_ex_0, r_ex_0
    for(int i = 0; i < q.coeffs().size(); i++) {
        q.coeffs()(i) = *(dev_ptr_set.q_ex_0 + i * num_proj_factors + idx_factor);
    }
    r = q.toRotationMatrix();
    for(int i = 0; i < r.size(); i++) {
        *(dev_ptr_set.r_ex_0 + i * num_proj_factors + idx_factor) = r(i);
    }
    for(int i = 0; i < p.size(); i++) {
        p(i) = *(dev_ptr_set.p_ex_0 + i * num_proj_factors + idx_factor);
    }
    pts = q * pts + p;      // pts_imu_i = q_ex_0 * pts_cam_i + p_ex_0
    for(int i = 0; i < pts.size(); i++) {
        *(dev_ptr_set.pts_imu_i + i * num_proj_factors + idx_factor) = pts(i);
    }
    r_trans = r.transpose();
    for(int i = 0; i < r_trans.size(); i++) {
        *(dev_ptr_set.r_ex_0_trans + i * num_proj_factors + idx_factor) = r_trans(i);
    }

    // q_ex_1, p_ex_1, r_ex_1
    for(int i = 0; i < q.coeffs().size(); i++) {
        q.coeffs()(i) = *(dev_ptr_set.q_ex_1 + i * num_proj_factors + idx_factor);
    }
    r = q.toRotationMatrix();
    for(int i = 0; i < r.size(); i++) {
        *(dev_ptr_set.r_ex_1 + i * num_proj_factors + idx_factor) = r(i);
    }
    for(int i = 0; i < p.size(); i++) {
        p(i) = *(dev_ptr_set.p_ex_1 + i * num_proj_factors + idx_factor);
    }
    pts = q.inverse() * (pts - p);  // pts_imu_j = pts_imu_i; pts_cam_j = q_ex_1.inverse() * (pts_imu_j - p_ex_1)
    for(int i = 0; i < pts.size(); i++) {
        *(dev_ptr_set.pts_cam_j + i * num_proj_factors + idx_factor) = pts(i);
    }
    r_trans = r.transpose();
    for(int i = 0; i < r_trans.size(); i++) {
        *(dev_ptr_set.r_ex_1_trans + i * num_proj_factors + idx_factor) = r_trans(i);
    }

    T dep_j = pts.z();  // pts = pts_cam_j;
    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    reduce(0, 0) = 1.0 / dep_j;
    reduce(0, 1) = 0.0;
    reduce(0, 2) = -pts(0) / (dep_j * dep_j);
    reduce(1, 0) = 0.0;
    reduce(1, 1) = 1.0 / dep_j;
    reduce(1, 2) = -pts(1) / (dep_j * dep_j);
    for(int i = 0; i < reduce.size(); i++) {
        *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor) = reduce(i);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> tmp_r_1;
    for(int i = 0; i < tmp_r_1.size(); i++) {
        tmp_r_1(i) = *(dev_ptr_set.r_ex_1_trans + i * num_proj_factors + idx_factor);
    }
    for(int i = 0; i < r.size(); i++) {
        r(i) = *(dev_ptr_set.r_ex_0 + i * num_proj_factors + idx_factor);
    }
    tmp_r_1 = tmp_r_1 * r;
    for(int i = 0; i < tmp_r_1.size(); i++) {
        *(dev_ptr_set.tmp_r_1 + i * num_proj_factors + idx_factor) = tmp_r_1(i);
    }

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_0_l(     // p_ex_0
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_ex_1_trans;
    for(int i = 0; i < r_ex_1_trans.size(); i++) {
        r_ex_1_trans(i) = *(dev_ptr_set.r_ex_1_trans + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> jacobian_0_l;
    T left_row[3];
    T right_col[3];
    for(int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            left_row[0] = reduce(row, 0);
            left_row[1] = reduce(row, 1);
            left_row[2] = reduce(row, 2);
            right_col[0] = r_ex_1_trans(0, col);
            right_col[1] = r_ex_1_trans(1, col);
            right_col[2] = r_ex_1_trans(2, col);
            jacobian_0_l(row, col) = left_row[0] * right_col[0] + left_row[1] * right_col[1] + left_row[2] * right_col[2];
        }
    }
    *(dev_ptr_set.jacobian_0 + 0 * num_proj_factors + idx_factor) = jacobian_0_l(0);
    *(dev_ptr_set.jacobian_0 + 1 * num_proj_factors + idx_factor) = jacobian_0_l(1);
    *(dev_ptr_set.jacobian_0 + 2 * num_proj_factors + idx_factor) = jacobian_0_l(2);
    *(dev_ptr_set.jacobian_0 + 6 * num_proj_factors + idx_factor) = jacobian_0_l(3);
    *(dev_ptr_set.jacobian_0 + 7 * num_proj_factors + idx_factor) = jacobian_0_l(4);
    *(dev_ptr_set.jacobian_0 + 8 * num_proj_factors + idx_factor) = jacobian_0_l(5);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_0_r(     // q_ex_0
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 3, 1> pts_cam_i;
    for(int i = 0; i < pts_cam_i.size(); i++) {
        pts_cam_i(i) = *(dev_ptr_set.pts_cam_i + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> tmp_r_1;
    for(int i = 0; i < tmp_r_1.size(); i++) {
        tmp_r_1(i) = *(dev_ptr_set.tmp_r_1 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_0_r_tmp = tmp_r_1 * (-UtilitySkewSymmetric<T>(pts_cam_i));
    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> jacobian_0_r;
    T left_row[3];
    T right_col[3];
    for(int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            left_row[0] = reduce(row, 0);
            left_row[1] = reduce(row, 1);
            left_row[2] = reduce(row, 2);
            right_col[0] = jacobian_0_r_tmp(0, col);
            right_col[1] = jacobian_0_r_tmp(1, col);
            right_col[2] = jacobian_0_r_tmp(2, col);
            jacobian_0_r(row, col) = left_row[0] * right_col[0] + left_row[1] * right_col[1] + left_row[2] * right_col[2];
        }
    }
    *(dev_ptr_set.jacobian_0 +  3 * num_proj_factors + idx_factor) = jacobian_0_r(0);
    *(dev_ptr_set.jacobian_0 +  4 * num_proj_factors + idx_factor) = jacobian_0_r(1);
    *(dev_ptr_set.jacobian_0 +  5 * num_proj_factors + idx_factor) = jacobian_0_r(2);
    *(dev_ptr_set.jacobian_0 +  9 * num_proj_factors + idx_factor) = jacobian_0_r(3);
    *(dev_ptr_set.jacobian_0 + 10 * num_proj_factors + idx_factor) = jacobian_0_r(4);
    *(dev_ptr_set.jacobian_0 + 11 * num_proj_factors + idx_factor) = jacobian_0_r(5);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_1_l(     // p_ex_1
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_ex_1_trans;
    for(int i = 0; i < r_ex_1_trans.size(); i++) {
        r_ex_1_trans(i) = *(dev_ptr_set.r_ex_1_trans + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> jacobian_1_l;
    T left_row[3];
    T right_col[3];
    for(int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            left_row[0] = reduce(row, 0);
            left_row[1] = reduce(row, 1);
            left_row[2] = reduce(row, 2);
            right_col[0] = r_ex_1_trans(0, col);
            right_col[1] = r_ex_1_trans(1, col);
            right_col[2] = r_ex_1_trans(2, col);
            jacobian_1_l(row, col) = -(left_row[0] * right_col[0] + left_row[1] * right_col[1] + left_row[2] * right_col[2]);
        }
    }
    *(dev_ptr_set.jacobian_1 + 0 * num_proj_factors + idx_factor) = jacobian_1_l(0);
    *(dev_ptr_set.jacobian_1 + 1 * num_proj_factors + idx_factor) = jacobian_1_l(1);
    *(dev_ptr_set.jacobian_1 + 2 * num_proj_factors + idx_factor) = jacobian_1_l(2);
    *(dev_ptr_set.jacobian_1 + 6 * num_proj_factors + idx_factor) = jacobian_1_l(3);
    *(dev_ptr_set.jacobian_1 + 7 * num_proj_factors + idx_factor) = jacobian_1_l(4);
    *(dev_ptr_set.jacobian_1 + 8 * num_proj_factors + idx_factor) = jacobian_1_l(5);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_1_r(     // q_ex_1
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 3, 1> pts_cam_j;
    for(int i = 0; i < pts_cam_j.size(); i++) {
        pts_cam_j(i) = *(dev_ptr_set.pts_cam_j + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_1_r_tmp = UtilitySkewSymmetric<T>(pts_cam_j);
    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> jacobian_1_r;
    T left_row[3];
    T right_col[3];
    for(int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            left_row[0] = reduce(row, 0);
            left_row[1] = reduce(row, 1);
            left_row[2] = reduce(row, 2);
            right_col[0] = jacobian_1_r_tmp(0, col);
            right_col[1] = jacobian_1_r_tmp(1, col);
            right_col[2] = jacobian_1_r_tmp(2, col);
            jacobian_1_r(row, col) = left_row[0] * right_col[0] + left_row[1] * right_col[1] + left_row[2] * right_col[2];
        }
    }
    *(dev_ptr_set.jacobian_1 +  3 * num_proj_factors + idx_factor) = jacobian_1_r(0);
    *(dev_ptr_set.jacobian_1 +  4 * num_proj_factors + idx_factor) = jacobian_1_r(1);
    *(dev_ptr_set.jacobian_1 +  5 * num_proj_factors + idx_factor) = jacobian_1_r(2);
    *(dev_ptr_set.jacobian_1 +  9 * num_proj_factors + idx_factor) = jacobian_1_r(3);
    *(dev_ptr_set.jacobian_1 + 10 * num_proj_factors + idx_factor) = jacobian_1_r(4);
    *(dev_ptr_set.jacobian_1 + 11 * num_proj_factors + idx_factor) = jacobian_1_r(5);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_2(       // inv_depth
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 3, 1> pts_i;
    for(int i = 0; i < pts_i.size(); i++) {
        pts_i(i) = *(dev_ptr_set.pts_i + i * num_proj_factors + idx_factor);
    }

    T inv_depth_i = *(dev_ptr_set.inv_depth + idx_factor);

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> tmp_r_1;
    for(int i = 0; i < tmp_r_1.size(); i++) {
        tmp_r_1(i) = *(dev_ptr_set.tmp_r_1 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 3, 1> jacobian_2_tmp = tmp_r_1 * pts_i * -1.0 / pow(inv_depth_i, 2);
    Eigen::Matrix<T, 2, 1> jacobian_2 = reduce * jacobian_2_tmp;
    *(dev_ptr_set.jacobian_2 + 0 * num_proj_factors + idx_factor) = jacobian_2(0);
    *(dev_ptr_set.jacobian_2 + 1 * num_proj_factors + idx_factor) = jacobian_2(1);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_jacobian_3(       // cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 3, 1> velocity_i;
    for(int i = 0; i < velocity_i.size(); i++) {
        velocity_i(i) = *(dev_ptr_set.velocity_i + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> velocity_j;
    for(int i = 0; i < velocity_j.size(); i++) {
        velocity_j(i) = *(dev_ptr_set.velocity_j + i * num_proj_factors + idx_factor);
    }

    T inv_depth_i = *(dev_ptr_set.inv_depth + idx_factor);

    Eigen::Matrix<T, 2, 3, Eigen::RowMajor> reduce;
    for(int i = 0; i < reduce.size(); i++) {
        reduce(i) = *(dev_ptr_set.reduce + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> tmp_r_1;
    for(int i = 0; i < tmp_r_1.size(); i++) {
        tmp_r_1(i) = *(dev_ptr_set.tmp_r_1 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 3, 1> jacobian_3_tmp = tmp_r_1 * velocity_i / inv_depth_i * -1.0;
    Eigen::Matrix<T, 2, 1> jacobian_3 = reduce * jacobian_3_tmp + velocity_j.head(2);
    *(dev_ptr_set.jacobian_3 + 0 * num_proj_factors + idx_factor) = jacobian_3(0);
    *(dev_ptr_set.jacobian_3 + 1 * num_proj_factors + idx_factor) = jacobian_3(1);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_residual(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 3, 1> pts_j_td;
    for (int i = 0; i < pts_j_td.size(); i++) {
        pts_j_td(i) = *(dev_ptr_set.pts_j_td + i * num_proj_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> pts_cam_j;
    for(int i = 0; i < pts_cam_j.size(); i++) {
        pts_cam_j(i) = *(dev_ptr_set.pts_cam_j + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    T dep_j = pts_cam_j.z();
    Eigen::Matrix<T, 2, 1> residual = (pts_cam_j / dep_j).template head<2>() - pts_j_td.template head<2>();
    *(dev_ptr_set.residual + 0 * num_proj_factors + idx_factor) = residual(0);
    *(dev_ptr_set.residual + 1 * num_proj_factors + idx_factor) = residual(1);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_robust_info(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    Eigen::Matrix<T, 2, 1> residual;
    for(int i = 0; i < residual.size(); i++) {
        residual(i) = *(dev_ptr_set.residual + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    Eigen::Matrix<T, 2, 2, Eigen::RowMajor> info = \
    Eigen::Matrix<T, 2, 2, Eigen::RowMajor>::Identity() * (FOCAL_LENGTH / 1.5) * (FOCAL_LENGTH / 1.5);

    Eigen::Matrix<T, 2, 1> weight_err = info * residual;
    T e2 = (residual.transpose() * weight_err)(0);
    Eigen::Matrix<T, 3, 1> rho;
    huber_loss<T>(1.0, e2, rho);
    // cauchy_loss(1.0, e2, rho);

    T robust_chi2 = rho[0];
    *(dev_ptr_set.robust_chi2 + idx_factor) = robust_chi2;

    Eigen::Matrix<T, 2, 2, Eigen::RowMajor> robust_info = rho[1] * info;
    if(rho[2] > 0.0) {
        robust_info += 2 * rho[2] * (weight_err * weight_err.transpose());
    }
    *(dev_ptr_set.robust_info + 0 * num_proj_factors + idx_factor) = robust_info(0);
    *(dev_ptr_set.robust_info + 1 * num_proj_factors + idx_factor) = robust_info(1);
    *(dev_ptr_set.robust_info + 2 * num_proj_factors + idx_factor) = robust_info(2);
    *(dev_ptr_set.robust_info + 3 * num_proj_factors + idx_factor) = robust_info(3);

    Eigen::Matrix<T, 2, 2> drho_info = rho[1] * info;  // T drho = rho[1];
    if(rho[2] > 0.0) {
        T D = 1.0 + 2.0 * e2 * rho[2] / rho[1];
        T sqrt_D = sqrt(D);
        T alpha_sq_norm = (1.0 - sqrt_D) / e2;
        drho_info -= (rho[1] * alpha_sq_norm) * (info * residual * residual.transpose() * info);
        drho_info *= (1 / sqrt_D);
    }
    Eigen::Matrix<T, 2, 1> drho_info_residual = drho_info * residual;
    *(dev_ptr_set.drho_info_residual + 0 * num_proj_factors + idx_factor) = drho_info_residual(0);
    *(dev_ptr_set.drho_info_residual + 1 * num_proj_factors + idx_factor) = drho_info_residual(1);

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_rhs_0(        // pose_ex_0
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T jacobian_0[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_0[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_0 + i * num_proj_factors + idx_factor);
    }

    T drho_info_residual[2];
    for(int i = 0; i < 2; i++) {
        drho_info_residual[i] = __ldg(dev_ptr_set.drho_info_residual + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    unsigned int row_start = dev_ptr_set.rhs_0_row_start[idx_factor];

    T left_row[2];
    for(int i = 0; i < 6; i++) {
        left_row[0] = jacobian_0[0][i];
        left_row[1] = jacobian_0[1][i];
        T src = -(left_row[0] * drho_info_residual[0] + left_row[1] * drho_info_residual[1]);
        T *dst_ptr = (Bpp + row_start + i);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_rhs_1(        // pose_ex_1
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T jacobian_1[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_1[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_1 + i * num_proj_factors + idx_factor);
    }

    T drho_info_residual[2];
    for(int i = 0; i < 2; i++) {
        drho_info_residual[i] = __ldg(dev_ptr_set.drho_info_residual + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    unsigned int row_start = dev_ptr_set.rhs_1_row_start[idx_factor];

    T left_row[2];
    for(int i = 0; i < 6; i++) {
        left_row[0] = jacobian_1[0][i];
        left_row[1] = jacobian_1[1][i];
        T src = -(left_row[0] * drho_info_residual[0] + left_row[1] * drho_info_residual[1]);
        T *dst_ptr = (Bpp + row_start + i);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_rhs_2(        // inv_depth
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bmm,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T jacobian_2[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_2[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_2 + i * num_proj_factors + idx_factor);
    }

    T drho_info_residual[2];
    for(int i = 0; i < 2; i++) {
        drho_info_residual[i] = __ldg(dev_ptr_set.drho_info_residual + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    unsigned int row_start = dev_ptr_set.rhs_2_row_start[idx_factor];

    T left_row[2];
    for(int i = 0; i < 1; i++) {
        left_row[0] = jacobian_2[0][i];
        left_row[1] = jacobian_2[1][i];
        T src = -(left_row[0] * drho_info_residual[0] + left_row[1] * drho_info_residual[1]);
        T *dst_ptr = (Bmm + row_start + i);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_rhs_3(        // cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T jacobian_3[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_3[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_3 + i * num_proj_factors + idx_factor);
    }

    T drho_info_residual[2];
    for(int i = 0; i < 2; i++) {
        drho_info_residual[i] = __ldg(dev_ptr_set.drho_info_residual + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : write outputs **********

    unsigned int row_start = dev_ptr_set.rhs_3_row_start[idx_factor];

    T left_row[2];
    for(int i = 0; i < 1; i++) {
        left_row[0] = jacobian_3[0][i];
        left_row[1] = jacobian_3[1][i];
        T src = -(left_row[0] * drho_info_residual[0] + left_row[1] * drho_info_residual[1]);
        T *dst_ptr = (Bpp + row_start + i);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_00(   // pose_ex_0, pose_ex_0
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_0[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_0[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_0 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_00_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_00_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 6; col++) {
            left_row[0] = jacobian_0[0][row];
            left_row[1] = jacobian_0[1][row];
            right_col[0] = jacobian_0[0][col];
            right_col[1] = jacobian_0[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_01(   // pose_ex_0, pose_ex_1
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_0[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_0[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_0 + i * num_proj_factors + idx_factor);
    }

    T jacobian_1[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_1[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_1 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_01_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_01_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_10_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_10_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 6; col++) {
            left_row[0] = jacobian_0[0][row];
            left_row[1] = jacobian_0[1][row];
            right_col[0] = jacobian_1[0][col];
            right_col[1] = jacobian_1[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_02(   // pose_ex_0, inv_depth
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_0[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_0[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_0 + i * num_proj_factors + idx_factor);
    }

    T jacobian_2[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_2[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_2 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_02_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_02_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_20_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_20_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_0[0][row];
            left_row[1] = jacobian_0[1][row];
            right_col[0] = jacobian_2[0][col];
            right_col[1] = jacobian_2[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpm + leading_dim_Hpm * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hmp + leading_dim_Hmp * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_03(   // pose_ex_0, cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_ex_0_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1)    ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_0[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_0[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_0 + i * num_proj_factors + idx_factor);
    }

    T jacobian_3[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_3[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_3 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_03_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_03_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_30_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_30_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_0[0][row];
            left_row[1] = jacobian_0[1][row];
            right_col[0] = jacobian_3[0][col];
            right_col[1] = jacobian_3[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_11(   // pose_ex_1, pose_ex_1
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_1[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_1[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_1 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_11_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_11_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 6; col++) {
            left_row[0] = jacobian_1[0][row];
            left_row[1] = jacobian_1[1][row];
            right_col[0] = jacobian_1[0][col];
            right_col[1] = jacobian_1[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_12(   // pose_ex_1, inv_depth
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_1[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_1[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_1 + i * num_proj_factors + idx_factor);
    }

    T jacobian_2[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_2[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_2 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_12_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_12_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_21_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_21_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_1[0][row];
            left_row[1] = jacobian_1[1][row];
            right_col[0] = jacobian_2[0][col];
            right_col[1] = jacobian_2[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpm + leading_dim_Hpm * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hmp + leading_dim_Hmp * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_13(   // pose_ex_1, cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_ex_1_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1)    ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_1[2][6];
    for(int i = 0; i < 12; i++) {
        jacobian_1[i / 6][i % 6] = __ldg(dev_ptr_set.jacobian_1 + i * num_proj_factors + idx_factor);
    }

    T jacobian_3[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_3[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_3 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_13_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_13_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_31_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_31_col_start[idx_factor];
    for(int row = 0; row < 6; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_1[0][row];
            left_row[1] = jacobian_1[1][row];
            right_col[0] = jacobian_3[0][col];
            right_col[1] = jacobian_3[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_22(   // inv_depth, inv_depth
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hmm_diag,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_2[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_2[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_2 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_22_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_22_col_start[idx_factor];
    for(int row = 0; row < 1; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_2[0][row];
            left_row[1] = jacobian_2[1][row];
            right_col[0] = jacobian_2[0][col];
            right_col[1] = jacobian_2[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hmm_diag + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_23(   // inv_depth, cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.inv_depth_is_fixed + idx_factor) == 1) 
            || (*(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1)    ) 
        {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_2[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_2[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_2 + i * num_proj_factors + idx_factor);
    }

    T jacobian_3[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_3[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_3 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_23_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_23_col_start[idx_factor];
    unsigned int row_start_trans = dev_ptr_set.hessian_32_row_start[idx_factor];
    unsigned int col_start_trans = dev_ptr_set.hessian_32_col_start[idx_factor];
    for(int row = 0; row < 1; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_2[0][row];
            left_row[1] = jacobian_2[1][row];
            right_col[0] = jacobian_3[0][col];
            right_col[1] = jacobian_3[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hmp + leading_dim_Hmp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
            if(src != 0.0) {
                T* dst_ptr = (Hpm + leading_dim_Hpm * (col_start_trans + row) + row_start_trans + col);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

template<typename T>
__global__ void proj_1f2c_hessian_33(   // cur_td, cur_td
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_proj_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.cur_td_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs **********

    T robust_info[2][2];
    for(int i = 0; i < 4; i++) {
        robust_info[i / 2][i % 2] = __ldg(dev_ptr_set.robust_info + i * num_proj_factors + idx_factor);
    }

    T jacobian_3[2][1];
    for(int i = 0; i < 2; i++) {
        jacobian_3[i / 1][i % 1] = __ldg(dev_ptr_set.jacobian_3 + i * num_proj_factors + idx_factor);
    }

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    T left_row[2];
    T right_col[2];
    T temp[2];
    row_start = dev_ptr_set.hessian_33_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_33_col_start[idx_factor];
    for(int row = 0; row < 1; row++) {
        for(int col = 0; col < 1; col++) {
            left_row[0] = jacobian_3[0][row];
            left_row[1] = jacobian_3[1][row];
            right_col[0] = jacobian_3[0][col];
            right_col[1] = jacobian_3[1][col];
            temp[0] = robust_info[0][0] * right_col[0] + robust_info[0][1] * right_col[1];
            temp[1] = robust_info[1][0] * right_col[0] + robust_info[1][1] * right_col[1];
            T src = left_row[0] * temp[0] + left_row[1] * temp[1];
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute and write outputs **********
}

// ------------------------------------------------------------------------------------------------------------------------

// instantiation for T = double

template __global__ void proj_1f2c_block_range<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_temp<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_0_l<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_0_r<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_1_l<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_1_r<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_2<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_3<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_residual<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_robust_info<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_rhs_0<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void proj_1f2c_rhs_1<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void proj_1f2c_rhs_2<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Bmm,
    bool marg
);

template __global__ void proj_1f2c_rhs_3<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_00<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_01<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_02<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpm,
    int leading_dim_Hpm,
    double* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_03<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_11<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_12<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpm,
    int leading_dim_Hpm,
    double* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_13<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_22<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hmm_diag,
    bool marg
);

template __global__ void proj_1f2c_hessian_23<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpm,
    int leading_dim_Hpm,
    double* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_33<double>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

// ------------------------------------------------------------------------------------------------------------------------

// instantiation for T = float

template __global__ void proj_1f2c_block_range<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_temp<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_0_l<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_0_r<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_1_l<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_1_r<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_2<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_jacobian_3<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_residual<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_robust_info<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void proj_1f2c_rhs_0<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void proj_1f2c_rhs_1<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void proj_1f2c_rhs_2<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Bmm,
    bool marg
);

template __global__ void proj_1f2c_rhs_3<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_00<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_01<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_02<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpm,
    int leading_dim_Hpm,
    float* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_03<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_11<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_12<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpm,
    int leading_dim_Hpm,
    float* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_13<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void proj_1f2c_hessian_22<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hmm_diag,
    bool marg
);

template __global__ void proj_1f2c_hessian_23<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpm,
    int leading_dim_Hpm,
    float* Hmp,
    int leading_dim_Hmp,
    bool marg
);

template __global__ void proj_1f2c_hessian_33<float>(
    int num_proj_factors,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

} // namespace VINS_FUSION_CUDA_BA

