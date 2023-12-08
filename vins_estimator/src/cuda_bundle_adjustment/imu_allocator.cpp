//
// Created by lmf on 23-7-22.
//

#include "imu_allocator.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void IMUFactorAllocator<T>::Alloc(const std::vector< SimpleIMUFactor<T> >& factor_vec) {
    num_factors = factor_vec.size();

    imu_idx_i.resize(num_factors, 1); imu_idx_i.setZero();
    imu_idx_j.resize(num_factors, 1); imu_idx_j.setZero();
    involved_in_marg.resize(num_factors, 1); involved_in_marg.setZero();

    pose_i_is_fixed.resize(num_factors, 1); pose_i_is_fixed.setZero();
    speed_bias_i_is_fixed.resize(num_factors, 1); speed_bias_i_is_fixed.setZero();
    pose_j_is_fixed.resize(num_factors, 1); pose_j_is_fixed.setZero();
    speed_bias_j_is_fixed.resize(num_factors, 1); speed_bias_j_is_fixed.setZero();

    p_i_v1.resize(3, num_factors); p_i_v1.setZero();
    q_i_v1.resize(4, num_factors); q_i_v1.setZero();
    v_i_v1.resize(3, num_factors); v_i_v1.setZero();
    bias_acc_i_v1.resize(3, num_factors); bias_acc_i_v1.setZero();
    bias_gyr_i_v1.resize(3, num_factors); bias_gyr_i_v1.setZero();

    p_j_v1.resize(3, num_factors); p_j_v1.setZero();
    q_j_v1.resize(4, num_factors); q_j_v1.setZero();
    v_j_v1.resize(3, num_factors); v_j_v1.setZero();
    bias_acc_j_v1.resize(3, num_factors); bias_acc_j_v1.setZero();
    bias_gyr_j_v1.resize(3, num_factors); bias_gyr_j_v1.setZero();

    sum_dt_v1.resize(num_factors, 1); sum_dt_v1.setZero();

    linearized_bias_acc_v1.resize(3, num_factors); linearized_bias_acc_v1.setZero();
    linearized_bias_gyr_v1.resize(3, num_factors); linearized_bias_gyr_v1.setZero();

    delta_p_v1.resize(3, num_factors); delta_p_v1.setZero();
    delta_q_v1.resize(4, num_factors); delta_q_v1.setZero();
    delta_v_v1.resize(3, num_factors); delta_v_v1.setZero();

    jacobian_p_bias_acc_v1.resize(9, num_factors); jacobian_p_bias_acc_v1.setZero();
    jacobian_p_bias_gyr_v1.resize(9, num_factors); jacobian_p_bias_gyr_v1.setZero();
    jacobian_q_bias_gyr_v1.resize(9, num_factors); jacobian_q_bias_gyr_v1.setZero();
    jacobian_v_bias_acc_v1.resize(9, num_factors); jacobian_v_bias_acc_v1.setZero();
    jacobian_v_bias_gyr_v1.resize(9, num_factors); jacobian_v_bias_gyr_v1.setZero();

    state_jacobian_v1.resize(225, num_factors); state_jacobian_v1.setZero();
    state_covariance_v1.resize(225, num_factors); state_covariance_v1.setZero();

    info_v1.resize(225, num_factors); info_v1.setZero();
    sqrt_info_v1.resize(225, num_factors); sqrt_info_v1.setZero();

    gravity_v1.resize(3, num_factors); gravity_v1.setZero();

    #pragma omp parallel for num_threads(numberOfCores)
    for(int i = 0; i < factor_vec.size(); i++) {
        const auto& factor = factor_vec[i];

        imu_idx_i(i) = factor.idx_i;
        imu_idx_j(i) = factor.idx_j;
        involved_in_marg(i) = factor.involved_in_marg;

        pose_i_is_fixed(i) = factor.pose_i_is_fixed ? 1 : 0;
        speed_bias_i_is_fixed(i) = factor.speed_bias_i_is_fixed ? 1 : 0;
        pose_j_is_fixed(i) = factor.pose_j_is_fixed ? 1 : 0;
        speed_bias_j_is_fixed(i) = factor.speed_bias_j_is_fixed ? 1 : 0;

        // ----------

        p_i_v1.col(i) = factor.p_i;
        q_i_v1.col(i) = factor.q_i.coeffs();
        v_i_v1.col(i) = factor.v_i;
        bias_acc_i_v1.col(i) = factor.bias_acc_i;
        bias_gyr_i_v1.col(i) = factor.bias_gyr_i;

        p_j_v1.col(i) = factor.p_j;
        q_j_v1.col(i) = factor.q_j.coeffs();
        v_j_v1.col(i) = factor.v_j;
        bias_acc_j_v1.col(i) = factor.bias_acc_j;
        bias_gyr_j_v1.col(i) = factor.bias_gyr_j;

        sum_dt_v1(i) = factor.pre_integration_j_sum_dt;

        linearized_bias_acc_v1.col(i) = factor.pre_integration_j_linearized_bias_acc;
        linearized_bias_gyr_v1.col(i) = factor.pre_integration_j_linearized_bias_gyr;

        delta_p_v1.col(i) = factor.pre_integration_j_delta_p;
        delta_q_v1.col(i) = factor.pre_integration_j_delta_q.coeffs();
        delta_v_v1.col(i) = factor.pre_integration_j_delta_v;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;

        temp = factor.jacobian_p_bias_acc; temp.resize(9,1); jacobian_p_bias_acc_v1.col(i) = temp;
        temp = factor.jacobian_p_bias_gyr; temp.resize(9,1); jacobian_p_bias_gyr_v1.col(i) = temp;
        temp = factor.jacobian_q_bias_gyr; temp.resize(9,1); jacobian_q_bias_gyr_v1.col(i) = temp;
        temp = factor.jacobian_v_bias_acc; temp.resize(9,1); jacobian_v_bias_acc_v1.col(i) = temp;
        temp = factor.jacobian_v_bias_gyr; temp.resize(9,1); jacobian_v_bias_gyr_v1.col(i) = temp;

        temp = factor.pre_integration_j_jacobian; temp.resize(225, 1); state_jacobian_v1.col(i) = temp;
        temp = factor.pre_integration_j_covariance; temp.resize(225, 1); state_covariance_v1.col(i) = temp;

        Eigen::Matrix<T, 15, 15, Eigen::RowMajor> temp_info = factor.pre_integration_j_covariance.inverse();
        Eigen::Matrix<T, 15, 15, Eigen::RowMajor> temp_sqrt_info = Eigen::LLT< Eigen::Matrix<T, 15, 15, Eigen::RowMajor> >(temp_info).matrixL().transpose();
        temp = temp_info; temp.resize(225, 1); info_v1.col(i) = temp;
        temp = temp_sqrt_info; temp.resize(225, 1); sqrt_info_v1.col(i) = temp;

        gravity_v1.col(i) = factor.gravity;
    }

    p_i_v2 = p_i_v1;
    q_i_v2 = q_i_v1;
    v_i_v2 = v_i_v1;
    bias_acc_i_v2 = bias_acc_i_v1;
    bias_gyr_i_v2 = bias_gyr_i_v1;

    p_j_v2 = p_j_v1;
    q_j_v2 = q_j_v1;
    v_j_v2 = v_j_v1;
    bias_acc_j_v2 = bias_acc_j_v1;
    bias_gyr_j_v2 = bias_gyr_j_v1;

    sum_dt_v2 = sum_dt_v1;

    linearized_bias_acc_v2 = linearized_bias_acc_v1;
    linearized_bias_gyr_v2 = linearized_bias_gyr_v1;

    delta_p_v2 = delta_p_v1;
    delta_q_v2 = delta_q_v1;
    delta_v_v2 = delta_v_v1;

    jacobian_p_bias_acc_v2 = jacobian_p_bias_acc_v1;
    jacobian_p_bias_gyr_v2 = jacobian_p_bias_gyr_v1;
    jacobian_q_bias_gyr_v2 = jacobian_q_bias_gyr_v1;
    jacobian_v_bias_acc_v2 = jacobian_v_bias_acc_v1;
    jacobian_v_bias_gyr_v2 = jacobian_v_bias_gyr_v1;

    state_jacobian_v2 = state_jacobian_v1;
    state_covariance_v2 = state_covariance_v1;

    info_v2 = info_v1;
    sqrt_info_v2 = sqrt_info_v1;

    gravity_v2 = gravity_v1;
}

// ----------

// instantiation
template class IMUFactorAllocator<double>;
template class IMUFactorAllocator<float>;

} // namespace VINS_FUSION_CUDA_BA
