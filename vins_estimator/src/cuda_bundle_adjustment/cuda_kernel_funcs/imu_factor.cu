//
// Created by lmf on 23-7-25.
//

#include "imu_factor.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void imu_block_range(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    int idx_i = *(dev_ptr_set.idx_i + idx_factor);
    int idx_j = *(dev_ptr_set.idx_j + idx_factor);

    unsigned int jacobian_0_col_start = 13 + idx_i * 15;
    unsigned int jacobian_0_col_end = jacobian_0_col_start + 6;
    unsigned int jacobian_1_col_start = jacobian_0_col_end;
    unsigned int jacobian_1_col_end = jacobian_1_col_start + 9;
    unsigned int jacobian_2_col_start = 13 + idx_j * 15;
    unsigned int jacobian_2_col_end = jacobian_2_col_start + 6;
    unsigned int jacobian_3_col_start = jacobian_2_col_end;
    unsigned int jacobian_3_col_end = jacobian_3_col_start + 9;

    BlockRange jacobian_0_block_range{0, 1, jacobian_0_col_start, jacobian_0_col_end};
    BlockRange jacobian_1_block_range{0, 1, jacobian_1_col_start, jacobian_1_col_end};
    BlockRange jacobian_2_block_range{0, 1, jacobian_2_col_start, jacobian_2_col_end};
    BlockRange jacobian_3_block_range{0, 1, jacobian_3_col_start, jacobian_3_col_end};

    BlockRange hessian_00_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_00_row_start + idx_factor) = hessian_00_block_range.row_start;
    *(dev_ptr_set.hessian_00_col_start + idx_factor) = hessian_00_block_range.col_start;
    BlockRange hessian_01_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_01_row_start + idx_factor) = hessian_01_block_range.row_start;
    *(dev_ptr_set.hessian_01_col_start + idx_factor) = hessian_01_block_range.col_start;
    BlockRange hessian_02_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_02_row_start + idx_factor) = hessian_02_block_range.row_start;
    *(dev_ptr_set.hessian_02_col_start + idx_factor) = hessian_02_block_range.col_start;
    BlockRange hessian_03_block_range = GetJTJBlockRange(jacobian_0_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_03_row_start + idx_factor) = hessian_03_block_range.row_start;
    *(dev_ptr_set.hessian_03_col_start + idx_factor) = hessian_03_block_range.col_start;

    BlockRange hessian_10_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_10_row_start + idx_factor) = hessian_10_block_range.row_start;
    *(dev_ptr_set.hessian_10_col_start + idx_factor) = hessian_10_block_range.col_start;
    BlockRange hessian_11_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_11_row_start + idx_factor) = hessian_11_block_range.row_start;
    *(dev_ptr_set.hessian_11_col_start + idx_factor) = hessian_11_block_range.col_start;
    BlockRange hessian_12_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_12_row_start + idx_factor) = hessian_12_block_range.row_start;
    *(dev_ptr_set.hessian_12_col_start + idx_factor) = hessian_12_block_range.col_start;
    BlockRange hessian_13_block_range = GetJTJBlockRange(jacobian_1_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_13_row_start + idx_factor) = hessian_13_block_range.row_start;
    *(dev_ptr_set.hessian_13_col_start + idx_factor) = hessian_13_block_range.col_start;

    BlockRange hessian_20_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_20_row_start + idx_factor) = hessian_20_block_range.row_start;
    *(dev_ptr_set.hessian_20_col_start + idx_factor) = hessian_20_block_range.col_start;
    BlockRange hessian_21_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_21_row_start + idx_factor) = hessian_21_block_range.row_start;
    *(dev_ptr_set.hessian_21_col_start + idx_factor) = hessian_21_block_range.col_start;
    BlockRange hessian_22_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_22_row_start + idx_factor) = hessian_22_block_range.row_start;
    *(dev_ptr_set.hessian_22_col_start + idx_factor) = hessian_22_block_range.col_start;
    BlockRange hessian_23_block_range = GetJTJBlockRange(jacobian_2_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_23_row_start + idx_factor) = hessian_23_block_range.row_start;
    *(dev_ptr_set.hessian_23_col_start + idx_factor) = hessian_23_block_range.col_start;

    BlockRange hessian_30_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_0_block_range);
    *(dev_ptr_set.hessian_30_row_start + idx_factor) = hessian_30_block_range.row_start;
    *(dev_ptr_set.hessian_30_col_start + idx_factor) = hessian_30_block_range.col_start;
    BlockRange hessian_31_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_1_block_range);
    *(dev_ptr_set.hessian_31_row_start + idx_factor) = hessian_31_block_range.row_start;
    *(dev_ptr_set.hessian_31_col_start + idx_factor) = hessian_31_block_range.col_start;
    BlockRange hessian_32_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_2_block_range);
    *(dev_ptr_set.hessian_32_row_start + idx_factor) = hessian_32_block_range.row_start;
    *(dev_ptr_set.hessian_32_col_start + idx_factor) = hessian_32_block_range.col_start;
    BlockRange hessian_33_block_range = GetJTJBlockRange(jacobian_3_block_range, jacobian_3_block_range);
    *(dev_ptr_set.hessian_33_row_start + idx_factor) = hessian_33_block_range.row_start;
    *(dev_ptr_set.hessian_33_col_start + idx_factor) = hessian_33_block_range.col_start;

    BlockRange rhs_0_block_range{jacobian_0_col_start, jacobian_0_col_end, 0, 1};
    *(dev_ptr_set.rhs_0_row_start + idx_factor) = rhs_0_block_range.row_start;
    BlockRange rhs_1_block_range{jacobian_1_col_start, jacobian_1_col_end, 0, 1};
    *(dev_ptr_set.rhs_1_row_start + idx_factor) = rhs_1_block_range.row_start;
    BlockRange rhs_2_block_range{jacobian_2_col_start, jacobian_2_col_end, 0, 1};
    *(dev_ptr_set.rhs_2_row_start + idx_factor) = rhs_2_block_range.row_start;
    BlockRange rhs_3_block_range{jacobian_3_col_start, jacobian_3_col_end, 0, 1};
    *(dev_ptr_set.rhs_3_row_start + idx_factor) = rhs_3_block_range.row_start;
}

template<typename T>
__global__ void imu_jacobian_0(     // pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 3, 1> p_i;
    for(int i = 0; i < p_i.size(); i++) {
        p_i(i) = *(dev_ptr_set.p_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_i;
    for(int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> v_i;
    for(int i = 0; i < v_i.size(); i++) {
        v_i(i) = *(dev_ptr_set.v_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    for(int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_i(i) = *(dev_ptr_set.bias_gyr_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> p_j;
    for(int i = 0; i < p_j.size(); i++) {
        p_j(i) = *(dev_ptr_set.p_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_j;
    for(int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(dev_ptr_set.q_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> v_j;
    for(int i = 0; i < v_j.size(); i++) {
        v_j(i) = *(dev_ptr_set.v_j + i * num_imu_factors + idx_factor);
    }

    T sum_dt = *(dev_ptr_set.sum_dt + idx_factor);

    Eigen::Matrix<T, 3, 1> linearized_bias_gyr;
    for(int i = 0; i < linearized_bias_gyr.size(); i++) {
        linearized_bias_gyr(i) = *(dev_ptr_set.linearized_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> delta_q;
    for(int i = 0; i < delta_q.coeffs().size(); i++) {
        delta_q.coeffs()(i) = *(dev_ptr_set.delta_q + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_q_bias_gyr;
    for(int i = 0; i < jacobian_q_bias_gyr.size(); i++) {
        jacobian_q_bias_gyr(i) = *(dev_ptr_set.jacobian_q_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> G;
    for(int i = 0; i < G.size(); i++) {
        G(i) = *(dev_ptr_set.gravity + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute jacobian_0 and write it to global memory **********

    Eigen::Quaternion<T> q_i_inv = q_i.inverse();
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_i_inv = q_i.inverse().toRotationMatrix();
    Eigen::Quaternion<T> corrected_delta_q = delta_q * UtilityDeltaQ<T>(jacobian_q_bias_gyr * (bias_gyr_i - linearized_bias_gyr));

    Eigen::Matrix<T, 4, 4, Eigen::RowMajor> temp = -( UtilityQLeft<T>(q_j.inverse() * q_i) * UtilityQRight<T>(corrected_delta_q) );
    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    jacobian_0.template block<3, 3>(O_P, O_P) = -r_i_inv;
    jacobian_0.template block<3, 3>(O_P, O_R) = UtilitySkewSymmetric<T>(q_i_inv * (0.5 * G * sum_dt * sum_dt + p_j - p_i - v_i * sum_dt));
    jacobian_0.template block<3, 3>(O_R, O_R) = temp.template bottomRightCorner<3, 3>();
    jacobian_0.template block<3, 3>(O_V, O_R) = UtilitySkewSymmetric<T>(q_i_inv * (G * sum_dt + v_j - v_i));
    for(int i = 0; i < jacobian_0.size(); i++) {
        if(jacobian_0(i) != 0.0) {
            *(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor) = jacobian_0(i);
        }
    }

    // ********** end : compute jacobian_0 and write it to global memory **********
}

template<typename T>
__global__ void imu_jacobian_1(     // speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Quaternion<T> q_i;
    for(int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    for(int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_i(i) = *(dev_ptr_set.bias_gyr_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_j;
    for(int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(dev_ptr_set.q_j + i * num_imu_factors + idx_factor);
    }

    T sum_dt = *(dev_ptr_set.sum_dt + idx_factor);

    Eigen::Matrix<T, 3, 1> linearized_bias_gyr;
    for(int i = 0; i < linearized_bias_gyr.size(); i++) {
        linearized_bias_gyr(i) = *(dev_ptr_set.linearized_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> delta_q;
    for(int i = 0; i < delta_q.coeffs().size(); i++) {
        delta_q.coeffs()(i) = *(dev_ptr_set.delta_q + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_acc;
    for(int i = 0; i < jacobian_p_bias_acc.size(); i++) {
        jacobian_p_bias_acc(i) = *(dev_ptr_set.jacobian_p_bias_acc + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_gyr;
    for(int i = 0; i < jacobian_p_bias_gyr.size(); i++) {
        jacobian_p_bias_gyr(i) = *(dev_ptr_set.jacobian_p_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_q_bias_gyr;
    for(int i = 0; i < jacobian_q_bias_gyr.size(); i++) {
        jacobian_q_bias_gyr(i) = *(dev_ptr_set.jacobian_q_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_acc;
    for(int i = 0; i < jacobian_v_bias_acc.size(); i++) {
        jacobian_v_bias_acc(i) = *(dev_ptr_set.jacobian_v_bias_acc + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_gyr;
    for(int i = 0; i < jacobian_v_bias_gyr.size(); i++) {
        jacobian_v_bias_gyr(i) = *(dev_ptr_set.jacobian_v_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_i_inv = q_i.inverse().toRotationMatrix();
    Eigen::Quaternion<T> q_j_inv = q_j.inverse();

    // ********** end : read inputs from global memory **********

    // ********** start : compute jacobian_1 and write it to global memory **********

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    jacobian_1.template block<3, 3>(O_P, O_V - O_V) = -r_i_inv * sum_dt;
    jacobian_1.template block<3, 3>(O_P, O_BA - O_V) = -jacobian_p_bias_acc;
    jacobian_1.template block<3, 3>(O_P, O_BG - O_V) = -jacobian_p_bias_gyr;
    jacobian_1.template block<3, 3>(O_R, O_BG - O_V) = -UtilityQLeft<T>(q_j_inv * q_i * delta_q).template bottomRightCorner<3, 3>() * jacobian_q_bias_gyr;
    jacobian_1.template block<3, 3>(O_V, O_V - O_V) = -r_i_inv;
    jacobian_1.template block<3, 3>(O_V, O_BA - O_V) = -jacobian_v_bias_acc;
    jacobian_1.template block<3, 3>(O_V, O_BG - O_V) = -jacobian_v_bias_gyr;
    jacobian_1.template block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity();
    jacobian_1.template block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity();
    for(int i = 0; i < jacobian_1.size(); i++) {
        if(jacobian_1(i) != 0.0) {
            *(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor) = jacobian_1(i);
        }
    }

    // ********** end : compute jacobian_1 and write it to global memory **********
}

template<typename T>
__global__ void imu_jacobian_2(     // pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Quaternion<T> q_i;
    for(int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    for(int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_i(i) = *(dev_ptr_set.bias_gyr_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_j;
    for(int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(dev_ptr_set.q_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> linearized_bias_gyr;
    for(int i = 0; i < linearized_bias_gyr.size(); i++) {
        linearized_bias_gyr(i) = *(dev_ptr_set.linearized_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> delta_q;
    for(int i = 0; i < delta_q.coeffs().size(); i++) {
        delta_q.coeffs()(i) = *(dev_ptr_set.delta_q + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_q_bias_gyr;
    for(int i = 0; i < jacobian_q_bias_gyr.size(); i++) {
        jacobian_q_bias_gyr(i) = *(dev_ptr_set.jacobian_q_bias_gyr + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute jacobian_2 and write it to global memory **********

    Eigen::Quaternion<T> q_i_inv = q_i.inverse();
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_i_inv = q_i.inverse().toRotationMatrix();
    Eigen::Quaternion<T> corrected_delta_q = delta_q * UtilityDeltaQ<T>(jacobian_q_bias_gyr * (bias_gyr_i - linearized_bias_gyr));

    // ********** jacobian_2 **********

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    jacobian_2.template block<3, 3>(O_P, O_P) = r_i_inv;
    jacobian_2.template block<3, 3>(O_R, O_R) = UtilityQLeft<T>(corrected_delta_q.inverse() * q_i_inv * q_j).template bottomRightCorner<3, 3>();
    for(int i = 0; i < jacobian_2.size(); i++) {
        if(jacobian_2(i) != 0.0) {
            *(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor) = jacobian_2(i);
        }
    }

    // ********** end : compute jacobian_2 and write it to global memory **********
}

template<typename T>
__global__ void imu_jacobian_3(     // speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Quaternion<T> q_i;
    for(int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute jacobian_3 and write it to global memory **********

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> r_i_inv = q_i.inverse().toRotationMatrix();

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    jacobian_3.template block<3, 3>(O_V, O_V - O_V) = r_i_inv;
    jacobian_3.template block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity();
    jacobian_3.template block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity();
    for(int i = 0; i < jacobian_3.size(); i++) {
        if(jacobian_3(i) != 0.0) {
            *(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor) = jacobian_3(i);
        }
    }

    // ********** end : compute jacobian_3 and write it to global memory **********
}

template<typename T>
__global__ void imu_residual(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 3, 1> p_i;
    for(int i = 0; i < p_i.size(); i++) {
        p_i(i) = *(dev_ptr_set.p_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_i;
    for(int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> v_i;
    for(int i = 0; i < v_i.size(); i++) {
        v_i(i) = *(dev_ptr_set.v_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_acc_i;
    for(int i = 0; i < bias_acc_i.size(); i++) {
        bias_acc_i(i) = *(dev_ptr_set.bias_acc_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    for(int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_i(i) = *(dev_ptr_set.bias_gyr_i + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> p_j;
    for(int i = 0; i < p_j.size(); i++) {
        p_j(i) = *(dev_ptr_set.p_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> q_j;
    for(int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(dev_ptr_set.q_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> v_j;
    for(int i = 0; i < v_j.size(); i++) {
        v_j(i) = *(dev_ptr_set.v_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_acc_j;
    for(int i = 0; i < bias_acc_i.size(); i++) {
        bias_acc_j(i) = *(dev_ptr_set.bias_acc_j + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_j;
    for(int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_j(i) = *(dev_ptr_set.bias_gyr_j + i * num_imu_factors + idx_factor);
    }

    T sum_dt = *(dev_ptr_set.sum_dt + idx_factor);

    Eigen::Matrix<T, 3, 1> linearized_bias_acc;
    for(int i = 0; i < linearized_bias_acc.size(); i++) {
        linearized_bias_acc(i) = *(dev_ptr_set.linearized_bias_acc + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> linearized_bias_gyr;
    for(int i = 0; i < linearized_bias_gyr.size(); i++) {
        linearized_bias_gyr(i) = *(dev_ptr_set.linearized_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> delta_p;
    for(int i = 0; i < delta_p.size(); i++) {
        delta_p(i) = *(dev_ptr_set.delta_p + i * num_imu_factors + idx_factor);
    }

    Eigen::Quaternion<T> delta_q;
    for(int i = 0; i < delta_q.coeffs().size(); i++) {
        delta_q.coeffs()(i) = *(dev_ptr_set.delta_q + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> delta_v;
    for(int i = 0; i < delta_v.size(); i++) {
        delta_v(i) = *(dev_ptr_set.delta_v + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_acc;
    for(int i = 0; i < jacobian_p_bias_acc.size(); i++) {
        jacobian_p_bias_acc(i) = *(dev_ptr_set.jacobian_p_bias_acc + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_gyr;
    for(int i = 0; i < jacobian_p_bias_gyr.size(); i++) {
        jacobian_p_bias_gyr(i) = *(dev_ptr_set.jacobian_p_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_q_bias_gyr;
    for(int i = 0; i < jacobian_q_bias_gyr.size(); i++) {
        jacobian_q_bias_gyr(i) = *(dev_ptr_set.jacobian_q_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_acc;
    for(int i = 0; i < jacobian_v_bias_acc.size(); i++) {
        jacobian_v_bias_acc(i) = *(dev_ptr_set.jacobian_v_bias_acc + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_gyr;
    for(int i = 0; i < jacobian_v_bias_gyr.size(); i++) {
        jacobian_v_bias_gyr(i) = *(dev_ptr_set.jacobian_v_bias_gyr + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 3, 1> G;
    for(int i = 0; i < G.size(); i++) {
        G(i) = *(dev_ptr_set.gravity + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute residual and write it to global memory **********

    Eigen::Matrix<T, 3, 1> delta_bias_acc = bias_acc_i - linearized_bias_acc;
    Eigen::Matrix<T, 3, 1> delta_bias_gyr = bias_gyr_i - linearized_bias_gyr;

    Eigen::Matrix<T, 3, 1> temp = (jacobian_q_bias_gyr * delta_bias_gyr) / T(2.0);
    Eigen::Quaternion<T> corrected_delta_q = delta_q * Eigen::Quaternion<T>{1.0, temp.x(), temp.y(), temp.z()};
    Eigen::Matrix<T, 3, 1> corrected_delta_v = delta_v;
    corrected_delta_v += (jacobian_v_bias_acc * delta_bias_acc);
    corrected_delta_v += (jacobian_v_bias_gyr * delta_bias_gyr);
    Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p;
    corrected_delta_p += (jacobian_p_bias_acc * delta_bias_acc);
    corrected_delta_p += (jacobian_p_bias_gyr * delta_bias_gyr);

    Eigen::Matrix<T, 15, 1> residual;
    residual.template block<3, 1>(O_P, 0) = q_i.inverse() * (0.5 * G * sum_dt * sum_dt + p_j - p_i - v_i * sum_dt) - corrected_delta_p;
    residual.template block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (q_i.inverse() * q_j)).vec();
    residual.template block<3, 1>(O_V, 0) = q_i.inverse() * (G * sum_dt + v_j - v_i) - corrected_delta_v;
    residual.template block<3, 1>(O_BA, 0) = bias_acc_j - bias_acc_i;
    residual.template block<3, 1>(O_BG, 0) = bias_gyr_j - bias_gyr_i;
    for(int i = 0; i < residual.size(); i++) {
        *(dev_ptr_set.residual + i * num_imu_factors + idx_factor) = residual(i);
    }

    // ********** end : compute residual and write it to global memory **********
}

template<typename T>
__global__ void imu_robust_info(
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> info;
    for(int r = 0; r < info.rows(); r++) {
        for(int c = r; c < info.cols(); c++) {
            unsigned int idx_1d = r * info.rows() + c;
            info(r, c) = *(dev_ptr_set.info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                info(c, r) = info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 1> residual;
    for(int i = 0; i < residual.size(); i++) {
        residual(i) = *(dev_ptr_set.residual + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute robust_chi2, robust_info, drho_info_residual and write them to global memory **********

    T robust_chi2 = (residual.transpose() * info * residual)(0);
    *(dev_ptr_set.robust_chi2 + idx_factor) = robust_chi2;

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info = info;
    for(int i = 0; i < robust_info.size(); i++) {
        *(dev_ptr_set.robust_info + i * num_imu_factors + idx_factor) = robust_info(i);
    }

    T drho = 1.0;

    Eigen::Matrix<T, 15, 1> drho_info_residual = drho * info * residual;
    for(int i = 0; i < drho_info_residual.size(); i++) {
        *(dev_ptr_set.drho_info_residual + i * num_imu_factors + idx_factor) = drho_info_residual(i);
    }

    // ********** end : compute robust_chi2, robust_info, drho_info_residual and write them to global memory **********
}

template<typename T>
__global__ void imu_rhs_0(      // pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    for(int i = 0; i < jacobian_0.size(); i++) {
        jacobian_0(i) = __ldg(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 1> drho_info_residual;
    for(int i = 0; i < drho_info_residual.size(); i++) {
        drho_info_residual(i) = __ldg(dev_ptr_set.drho_info_residual + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute rhs_0 and write it to global memory **********

    unsigned int row_start = 0;

    Eigen::Matrix<T, 6, 1> rhs_0 = jacobian_0.transpose() * drho_info_residual;
    row_start = dev_ptr_set.rhs_0_row_start[idx_factor];
    for(int row = 0; row < rhs_0.rows(); row++) {
        T *dst_ptr = (Bpp + row_start + row);
        T src = -rhs_0(row);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : compute rhs_0 and write it to global memory **********
}

template<typename T>
__global__ void imu_rhs_1(      // speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    for(int i = 0; i < jacobian_1.size(); i++) {
        jacobian_1(i) = __ldg(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 1> drho_info_residual;
    for(int i = 0; i < drho_info_residual.size(); i++) {
        drho_info_residual(i) = __ldg(dev_ptr_set.drho_info_residual + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute rhs_1 and write it to global memory **********

    unsigned int row_start = 0;

    Eigen::Matrix<T, 9, 1> rhs_1 = jacobian_1.transpose() * drho_info_residual;
    row_start = dev_ptr_set.rhs_1_row_start[idx_factor];
    for(int row = 0; row < rhs_1.rows(); row++) {
        T *dst_ptr = (Bpp + row_start + row);
        T src = -rhs_1(row);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : compute rhs_1 and write it to global memory **********
}

template<typename T>
__global__ void imu_rhs_2(      // pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    for(int i = 0; i < jacobian_2.size(); i++) {
        jacobian_2(i) = __ldg(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 1> drho_info_residual;
    for(int i = 0; i < drho_info_residual.size(); i++) {
        drho_info_residual(i) = __ldg(dev_ptr_set.drho_info_residual + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute rhs_2 and write it to global memory **********

    unsigned int row_start = 0;

    Eigen::Matrix<T, 6, 1> rhs_2 = jacobian_2.transpose() * drho_info_residual;
    row_start = dev_ptr_set.rhs_2_row_start[idx_factor];
    for(int row = 0; row < rhs_2.rows(); row++) {
        T *dst_ptr = (Bpp + row_start + row);
        T src = -rhs_2(row);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : compute rhs_2 and write it to global memory **********
}

template<typename T>
__global__ void imu_rhs_3(      // speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Bpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    for(int i = 0; i < jacobian_3.size(); i++) {
        jacobian_3(i) = __ldg(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 1> drho_info_residual;
    for(int i = 0; i < drho_info_residual.size(); i++) {
        drho_info_residual(i) = __ldg(dev_ptr_set.drho_info_residual + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute rhs_3 and write it to global memory **********

    unsigned int row_start = 0;

    Eigen::Matrix<T, 9, 1> rhs_3 = jacobian_3.transpose() * drho_info_residual;
    row_start = dev_ptr_set.rhs_3_row_start[idx_factor];
    for(int row = 0; row < rhs_3.rows(); row++) {
        T *dst_ptr = (Bpp + row_start + row);
        T src = -rhs_3(row, 0);
        MyAtomicAdd<T>(dst_ptr, src);
    }

    // ********** end : compute rhs_3 and write it to global memory **********
}

template<typename T>
__global__ void imu_hessian_00(     // pose_i, pose_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    for(int i = 0; i < jacobian_0.size(); i++) {
        jacobian_0(i) = __ldg(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_00 and write it to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 6, Eigen::RowMajor> hessian_00 = jacobian_0.transpose() * robust_info * jacobian_0;
    row_start = dev_ptr_set.hessian_00_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_00_col_start[idx_factor];
    for(int row = 0; row < hessian_00.rows(); row++) {
        for(int col = 0; col < hessian_00.cols(); col++) {
            T src = hessian_00(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_00 and write it to global memory **********
}

template<typename T>
__global__ void imu_hessian_01(     // pose_i, speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    for(int i = 0; i < jacobian_0.size(); i++) {
        jacobian_0(i) = __ldg(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    for(int i = 0; i < jacobian_1.size(); i++) {
        jacobian_1(i) = __ldg(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_01 and hessian_01 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 9, Eigen::RowMajor> hessian_01 = jacobian_0.transpose() * robust_info * jacobian_1;
    row_start = dev_ptr_set.hessian_01_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_01_col_start[idx_factor];
    for(int row = 0; row < hessian_01.rows(); row++) {
        for(int col = 0; col < hessian_01.cols(); col++) {
            T src = hessian_01(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_10_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_10_col_start[idx_factor];
    for(int row = 0; row < hessian_01.cols(); row++) {
        for(int col = 0; col < hessian_01.rows(); col++) {
            T src = hessian_01(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_01 and hessian_01 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_02(     // pose_i, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    for(int i = 0; i < jacobian_0.size(); i++) {
        jacobian_0(i) = __ldg(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    for(int i = 0; i < jacobian_2.size(); i++) {
        jacobian_2(i) = __ldg(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_02 and hessian_20 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 6, Eigen::RowMajor> hessian_02 = jacobian_0.transpose() * robust_info * jacobian_2;
    row_start = dev_ptr_set.hessian_02_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_02_col_start[idx_factor];
    for(int row = 0; row < hessian_02.rows(); row++) {
        for(int col = 0; col < hessian_02.cols(); col++) {
            T src = hessian_02(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_20_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_20_col_start[idx_factor];
    for(int row = 0; row < hessian_02.cols(); row++) {
        for(int col = 0; col < hessian_02.rows(); col++) {
            T src = hessian_02(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_02 and hessian_20 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_03(     // pose_i, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_i_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);     // *(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_0;
    jacobian_0.setZero();
    for(int i = 0; i < jacobian_0.size(); i++) {
        jacobian_0(i) = __ldg(dev_ptr_set.jacobian_0 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    for(int i = 0; i < jacobian_3.size(); i++) {
        jacobian_3(i) = __ldg(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_03 and hessian_30 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 9, Eigen::RowMajor> hessian_03 = jacobian_0.transpose() * robust_info * jacobian_3;
    row_start = dev_ptr_set.hessian_03_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_03_col_start[idx_factor];
    for(int row = 0; row < hessian_03.rows(); row++) {
        for(int col = 0; col < hessian_03.cols(); col++) {
            T src = hessian_03(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_30_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_30_col_start[idx_factor];
    for(int row = 0; row < hessian_03.cols(); row++) {
        for(int col = 0; col < hessian_03.rows(); col++) {
            T src = hessian_03(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_03 and hessian_30 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_11(     // speed_bias_i, speed_bias_i
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    for(int i = 0; i < jacobian_1.size(); i++) {
        jacobian_1(i) = __ldg(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_11 and write it to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 9, 9, Eigen::RowMajor> hessian_11 = jacobian_1.transpose() * robust_info * jacobian_1;
    row_start = dev_ptr_set.hessian_11_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_11_col_start[idx_factor];
    for(int row = 0; row < hessian_11.rows(); row++) {
        for(int col = 0; col < hessian_11.cols(); col++) {
            T src = hessian_11(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_11 and write it to global memory **********
}

template<typename T>
__global__ void imu_hessian_12(     // speed_bias_i, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1)       ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    for(int i = 0; i < jacobian_1.size(); i++) {
        jacobian_1(i) = __ldg(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    for(int i = 0; i < jacobian_2.size(); i++) {
        jacobian_2(i) = __ldg(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_12 and hessian_21 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 9, 6, Eigen::RowMajor> hessian_12 = jacobian_1.transpose() * robust_info * jacobian_2;
    row_start = dev_ptr_set.hessian_12_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_12_col_start[idx_factor];
    for(int row = 0; row < hessian_12.rows(); row++) {
        for(int col = 0; col < hessian_12.cols(); col++) {
            T src = hessian_12(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_21_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_21_col_start[idx_factor];
    for(int row = 0; row < hessian_12.cols(); row++) {
        for(int col = 0; col < hessian_12.rows(); col++) {
            T src = hessian_12(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_12 and hessian_21 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_13(     // speed_bias_i, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.speed_bias_i_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_1;
    jacobian_1.setZero();
    for(int i = 0; i < jacobian_1.size(); i++) {
        jacobian_1(i) = __ldg(dev_ptr_set.jacobian_1 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    for(int i = 0; i < jacobian_3.size(); i++) {
        jacobian_3(i) = __ldg(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_13 and hessian_31 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 9, 9, Eigen::RowMajor> hessian_13 = jacobian_1.transpose() * robust_info * jacobian_3;
    row_start = dev_ptr_set.hessian_13_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_13_col_start[idx_factor];
    for(int row = 0; row < hessian_13.rows(); row++) {
        for(int col = 0; col < hessian_13.cols(); col++) {
            T src = hessian_13(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_31_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_31_col_start[idx_factor];
    for(int row = 0; row < hessian_13.cols(); row++) {
        for(int col = 0; col < hessian_13.rows(); col++) {
            T src = hessian_13(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_13 and hessian_31 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_22(     // pose_j, pose_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    for(int i = 0; i < jacobian_2.size(); i++) {
        jacobian_2(i) = __ldg(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_22 and write it to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 6, Eigen::RowMajor> hessian_22 = jacobian_2.transpose() * robust_info * jacobian_2;
    row_start = dev_ptr_set.hessian_22_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_22_col_start[idx_factor];
    for(int row = 0; row < hessian_22.rows(); row++) {
        for(int col = 0; col < hessian_22.cols(); col++) {
            T src = hessian_22(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_22 and write it to global memory **********
}

template<typename T>
__global__ void imu_hessian_23(     // pose_j, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if(    (*(dev_ptr_set.pose_j_is_fixed + idx_factor) == 1)
            || (*(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1) ) 
        {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 6, Eigen::RowMajor> jacobian_2;
    jacobian_2.setZero();
    for(int i = 0; i < jacobian_2.size(); i++) {
        jacobian_2(i) = __ldg(dev_ptr_set.jacobian_2 + i * num_imu_factors + idx_factor);
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    for(int i = 0; i < jacobian_3.size(); i++) {
        jacobian_3(i) = __ldg(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_23 and hessian_32 and write them to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 6, 9, Eigen::RowMajor> hessian_23 = jacobian_2.transpose() * robust_info * jacobian_3;
    row_start = dev_ptr_set.hessian_23_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_23_col_start[idx_factor];
    for(int row = 0; row < hessian_23.rows(); row++) {
        for(int col = 0; col < hessian_23.cols(); col++) {
            T src = hessian_23(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    row_start = dev_ptr_set.hessian_32_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_32_col_start[idx_factor];
    for(int row = 0; row < hessian_23.cols(); row++) {
        for(int col = 0; col < hessian_23.rows(); col++) {
            T src = hessian_23(col, row);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_23 and hessian_32 and write them to global memory **********
}

template<typename T>
__global__ void imu_hessian_33(     // speed_bias_j, speed_bias_j
    int num_imu_factors,
    IMUFactorDevPtrSet<T> dev_ptr_set,
    T* Hpp,
    int leading_dim_Hpp,
    bool marg
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_factor >= num_imu_factors)
        return;

    if(marg) {
        if( *(dev_ptr_set.involved_in_marg + idx_factor) != 1 ) {
            return;
        }
    }

    if(!marg) {     // if param is fixed, then return without any computation;
        if( *(dev_ptr_set.speed_bias_j_is_fixed + idx_factor) == 1 ) {
            return;
        }
    }

    // ********** start : read inputs from global memory **********

    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> robust_info;
    for(int r = 0; r < robust_info.rows(); r++) {
        for(int c = r; c < robust_info.cols(); c++) {
            unsigned int idx_1d = r * robust_info.rows() + c;
            robust_info(r, c) = __ldg(dev_ptr_set.robust_info + idx_1d * num_imu_factors + idx_factor);
            if(c > r) {
                robust_info(c, r) = robust_info(r, c);
            }
        }
    }

    Eigen::Matrix<T, 15, 9, Eigen::RowMajor> jacobian_3;
    jacobian_3.setZero();
    for(int i = 0; i < jacobian_3.size(); i++) {
        jacobian_3(i) = __ldg(dev_ptr_set.jacobian_3 + i * num_imu_factors + idx_factor);
    }

    // ********** end : read inputs from global memory **********

    // ********** start : compute hessian_33 and write it to global memory **********

    unsigned int row_start = 0;
    unsigned int col_start = 0;

    Eigen::Matrix<T, 9, 9, Eigen::RowMajor> hessian_33 = jacobian_3.transpose() * robust_info * jacobian_3;
    row_start = dev_ptr_set.hessian_33_row_start[idx_factor];
    col_start = dev_ptr_set.hessian_33_col_start[idx_factor];
    for(int row = 0; row < hessian_33.rows(); row++) {
        for(int col = 0; col < hessian_33.cols(); col++) {
            T src = hessian_33(row, col);
            if(src != 0.0) {
                T* dst_ptr = (Hpp + leading_dim_Hpp * (col_start + col) + row_start + row);
                MyAtomicAdd<T>(dst_ptr, src);
            }
        }
    }

    // ********** end : compute hessian_33 and write it to global memory **********
}

// ------------------------------------------------------------------------------------------------------------------------

// instantiation for T = double

template __global__ void imu_block_range<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set
);

template __global__ void imu_jacobian_0<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_1<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_2<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_3<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_residual<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_robust_info<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    bool marg
);

template __global__ void imu_rhs_0<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void imu_rhs_1<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void imu_rhs_2<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void imu_rhs_3<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Bpp,
    bool marg
);

template __global__ void imu_hessian_00<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_01<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_02<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_03<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_11<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_12<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_13<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_22<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_23<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_33<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double> dev_ptr_set,
    double* Hpp,
    int leading_dim_Hpp,
    bool marg
);

// ------------------------------------------------------------------------------------------------------------------------

// instantiation for T = float

template __global__ void imu_block_range<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set
);

template __global__ void imu_jacobian_0<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_1<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_2<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_jacobian_3<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_residual<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_robust_info<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    bool marg
);

template __global__ void imu_rhs_0<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void imu_rhs_1<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void imu_rhs_2<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void imu_rhs_3<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Bpp,
    bool marg
);

template __global__ void imu_hessian_00<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_01<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_02<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_03<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_11<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_12<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_13<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_22<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_23<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

template __global__ void imu_hessian_33<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float> dev_ptr_set,
    float* Hpp,
    int leading_dim_Hpp,
    bool marg
);

} // namespace VINS_FUSION_CUDA_BA



