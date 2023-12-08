//
// Created by lmf on 23-7-20.
//

#ifndef CUDA_BA_FACTORS_H
#define CUDA_BA_FACTORS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <cassert>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "common.h"

#include "../factor/integration_base.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class SimpleIMUFactor {

public : 
    int idx_i = 0;
    int idx_j = 0;

    char involved_in_marg = 0;

    Eigen::Matrix<T, 3, 1> gravity;

    Eigen::Matrix<T, 3, 1> p_i;
    Eigen::Quaternion<T> q_i;
    Eigen::Matrix<T, 3, 1> v_i;
    Eigen::Matrix<T, 3, 1> bias_acc_i;
    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    Eigen::Matrix<T, 3, 1> p_j;
    Eigen::Quaternion<T> q_j;
    Eigen::Matrix<T, 3, 1> v_j;
    Eigen::Matrix<T, 3, 1> bias_acc_j;
    Eigen::Matrix<T, 3, 1> bias_gyr_j;

    T pre_integration_j_sum_dt{};
    Eigen::Matrix<T, 3, 1> pre_integration_j_linearized_bias_acc;
    Eigen::Matrix<T, 3, 1> pre_integration_j_linearized_bias_gyr;
    Eigen::Matrix<T, 3, 1> pre_integration_j_delta_p;
    Eigen::Quaternion<T> pre_integration_j_delta_q;
    Eigen::Matrix<T, 3, 1> pre_integration_j_delta_v;
    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> pre_integration_j_jacobian;
    Eigen::Matrix<T, 15, 15, Eigen::RowMajor> pre_integration_j_covariance;

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_acc;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_p_bias_gyr;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_q_bias_gyr;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_acc;
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> jacobian_v_bias_gyr;

    bool pose_i_is_fixed;
    bool speed_bias_i_is_fixed;
    bool pose_j_is_fixed;
    bool speed_bias_j_is_fixed;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public : 
    explicit SimpleIMUFactor(
        int input_idx_i, 
        int input_idx_j, 
        Eigen::Matrix<double, 7, 1>& input_Pose_i,
        Eigen::Matrix<double, 9, 1>& input_SpeedBias_i,
        Eigen::Matrix<double, 7, 1>& input_Pose_j,
        Eigen::Matrix<double, 9, 1>& input_SpeedBias_j,
        Eigen::Vector3d& input_gravity,
        IntegrationBase* input_preinteg_ptr,
        bool input_pose_i_is_fixed,
        bool input_speed_bias_i_is_fixed,
        bool input_pose_j_is_fixed,
        bool input_speed_bias_j_is_fixed
    );
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
class SimpleProj2F2CFactor {

public : 
    int idx_i = 0;
    int idx_j = 0;

    int feature_index = 0;
    int feature_index_for_marg = -1;
    char involved_in_marg = 0;

    Eigen::Matrix<T, 3, 1> pts_i;
    Eigen::Matrix<T, 3, 1> pts_j;

    Eigen::Matrix<T, 3, 1> velocity_i;
    Eigen::Matrix<T, 3, 1> velocity_j;

    T td_i;
    T td_j;

    Eigen::Matrix<T, 3, 1> p_i;
    Eigen::Quaternion<T> q_i;
    Eigen::Matrix<T, 3, 1> p_j;
    Eigen::Quaternion<T> q_j;
    Eigen::Matrix<T, 3, 1> p_ex_0;
    Eigen::Quaternion<T> q_ex_0;
    Eigen::Matrix<T, 3, 1> p_ex_1;
    Eigen::Quaternion<T> q_ex_1;
    T inv_depth;
    T time_delay;

    bool pose_i_is_fixed;
    bool pose_j_is_fixed;
    bool pose_ex_0_is_fixed;
    bool pose_ex_1_is_fixed;
    bool inv_depth_is_fixed;
    bool cur_td_is_fixed;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public : 
    explicit SimpleProj2F2CFactor(
        int input_idx_i,
        int input_idx_j,
        int input_feature_index,
        Eigen::Vector3d& input_pts_i,
        Eigen::Vector3d& input_pts_j,
        Eigen::Vector2d& input_velocity_i,
        Eigen::Vector2d& input_velocity_j,
        double input_td_i,
        double input_td_j,
        Eigen::Matrix<double, 7, 1>& input_Pose_i,
        Eigen::Matrix<double, 7, 1>& input_Pose_j,
        Eigen::Matrix<double, 7, 1>& input_Ex_Pose_0,
        Eigen::Matrix<double, 7, 1>& input_Ex_Pose_1,
        double input_inv_depth,
        double input_time_delay,
        bool input_pose_i_is_fixed,
        bool input_pose_j_is_fixed,
        bool input_pose_ex_0_is_fixed,
        bool input_pose_ex_1_is_fixed,
        bool input_inv_depth_is_fixed,
        bool input_cur_td_is_fixed
    );
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
class SimpleProj2F1CFactor {

public : 
    int idx_i;
    int idx_j;

    int feature_index;
    int feature_index_for_marg;
    char involved_in_marg;

    Eigen::Matrix<T, 3, 1> pts_i;
    Eigen::Matrix<T, 3, 1> pts_j;

    Eigen::Matrix<T, 3, 1> velocity_i;
    Eigen::Matrix<T, 3, 1> velocity_j;

    T td_i;
    T td_j;

    Eigen::Matrix<T, 3, 1> p_i;
    Eigen::Quaternion<T> q_i;
    Eigen::Matrix<T, 3, 1> p_j;
    Eigen::Quaternion<T> q_j;
    Eigen::Matrix<T, 3, 1> p_ex_0;
    Eigen::Quaternion<T> q_ex_0;
    T inv_depth;
    T time_delay;

    bool pose_i_is_fixed;
    bool pose_j_is_fixed;
    bool pose_ex_0_is_fixed;
    bool inv_depth_is_fixed;
    bool cur_td_is_fixed;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public : 
    explicit SimpleProj2F1CFactor(
        int input_idx_i,
        int input_idx_j,
        int input_feature_index,
        Eigen::Vector3d& input_pts_i,
        Eigen::Vector3d& input_pts_j,
        Eigen::Vector2d& input_velocity_i,
        Eigen::Vector2d& input_velocity_j,
        double input_td_i,
        double input_td_j,
        Eigen::Matrix<double, 7, 1>& input_Pose_i,
        Eigen::Matrix<double, 7, 1>& input_Pose_j,
        Eigen::Matrix<double, 7, 1>& input_Ex_Pose_0,
        double input_inv_depth,
        double input_time_delay,
        bool input_pose_i_is_fixed,
        bool input_pose_j_is_fixed,
        bool input_pose_ex_0_is_fixed,
        bool input_inv_depth_is_fixed,
        bool input_cur_td_is_fixed
    );
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
class SimpleProj1F2CFactor {

public : 
    int idx_i = 0;
    int idx_j = 0;

    int feature_index = 0;
    int feature_index_for_marg = -1;
    char involved_in_marg = 0;

    Eigen::Matrix<T, 3, 1> pts_i;
    Eigen::Matrix<T, 3, 1> pts_j;

    Eigen::Matrix<T, 3, 1> velocity_i;
    Eigen::Matrix<T, 3, 1> velocity_j;

    T td_i;
    T td_j;

    Eigen::Matrix<T, 3, 1> p_ex_0;
    Eigen::Quaternion<T> q_ex_0;
    Eigen::Matrix<T, 3, 1> p_ex_1;
    Eigen::Quaternion<T> q_ex_1;
    T inv_depth;
    T time_delay;

    bool pose_ex_0_is_fixed;
    bool pose_ex_1_is_fixed;
    bool inv_depth_is_fixed;
    bool cur_td_is_fixed;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public : 
    explicit SimpleProj1F2CFactor(
        int input_idx_i,
        int input_idx_j,
        int input_feature_index,
        Eigen::Vector3d& input_pts_i,
        Eigen::Vector3d& input_pts_j,
        Eigen::Vector2d& input_velocity_i,
        Eigen::Vector2d& input_velocity_j,
        double input_td_i,
        double input_td_j,
        Eigen::Matrix<double, 7, 1>& input_Ex_Pose_0,
        Eigen::Matrix<double, 7, 1>& input_Ex_Pose_1,
        double input_inv_depth,
        double input_time_delay,
        bool input_pose_ex_0_is_fixed,
        bool input_pose_ex_1_is_fixed,
        bool input_inv_depth_is_fixed,
        bool input_cur_td_is_fixed
    );
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_FACTORS_H
