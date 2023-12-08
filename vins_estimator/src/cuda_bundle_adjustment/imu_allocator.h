//
// Created by lmf on 23-7-21.
//

#ifndef CUDA_BA_IMU_ALLOCATOR_H
#define CUDA_BA_IMU_ALLOCATOR_H

#include "factors.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class IMUFactorAllocator {

public :

    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_i;
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_j;

    Eigen::Matrix<char, Eigen::Dynamic, 1> involved_in_marg;

    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_i_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> speed_bias_i_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_j_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> speed_bias_j_is_fixed;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> v_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> bias_acc_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> bias_gyr_i_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> v_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> bias_acc_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> bias_gyr_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, 1> sum_dt_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> linearized_bias_acc_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> linearized_bias_gyr_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> delta_p_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> delta_q_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> delta_v_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobian_p_bias_acc_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobian_p_bias_gyr_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobian_q_bias_gyr_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobian_v_bias_acc_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobian_v_bias_gyr_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> state_jacobian_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> state_covariance_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> info_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> sqrt_info_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> gravity_v1;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bias_acc_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bias_gyr_i_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bias_acc_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bias_gyr_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, 1>  sum_dt_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> linearized_bias_acc_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> linearized_bias_gyr_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> delta_p_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> delta_q_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> delta_v_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_p_bias_acc_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_p_bias_gyr_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_q_bias_gyr_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_v_bias_acc_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian_v_bias_gyr_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> state_jacobian_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> state_covariance_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> info_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sqrt_info_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gravity_v2;

public :
    IMUFactorAllocator() = default;

public :
    size_t num_factors = 0;

public :
    void Alloc(const std::vector< SimpleIMUFactor<T> >& input_vec);
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_IMU_ALLOCATOR_H
