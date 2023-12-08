//
// Created by lmf on 23-7-20.
//

#include "factors.h"

namespace VINS_FUSION_CUDA_BA {

template<>
SimpleIMUFactor<double>::SimpleIMUFactor(
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
)
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    involved_in_marg(0),
    gravity(input_gravity),
    pre_integration_j_sum_dt(input_preinteg_ptr->sum_dt),
    pre_integration_j_linearized_bias_acc(input_preinteg_ptr->linearized_ba),
    pre_integration_j_linearized_bias_gyr(input_preinteg_ptr->linearized_bg),
    pre_integration_j_delta_p(input_preinteg_ptr->delta_p),
    pre_integration_j_delta_q(input_preinteg_ptr->delta_q),
    pre_integration_j_delta_v(input_preinteg_ptr->delta_v),
    pre_integration_j_jacobian(input_preinteg_ptr->jacobian),
    pre_integration_j_covariance(input_preinteg_ptr->covariance),
    pose_i_is_fixed(input_pose_i_is_fixed),
    speed_bias_i_is_fixed(input_speed_bias_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    speed_bias_j_is_fixed(input_speed_bias_j_is_fixed)
{
    jacobian_p_bias_acc = pre_integration_j_jacobian.block<3, 3>(O_P, O_BA);
    jacobian_p_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_P, O_BG);
    jacobian_q_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_R, O_BG);
    jacobian_v_bias_acc = pre_integration_j_jacobian.block<3, 3>(O_V, O_BA);
    jacobian_v_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_V, O_BG);

    p_i << input_Pose_i[0], input_Pose_i[1], input_Pose_i[2];
    q_i.coeffs() << input_Pose_i[3], input_Pose_i[4], input_Pose_i[5], input_Pose_i[6];
    v_i << input_SpeedBias_i[0], input_SpeedBias_i[1], input_SpeedBias_i[2];
    bias_acc_i << input_SpeedBias_i[3], input_SpeedBias_i[4], input_SpeedBias_i[5];
    bias_gyr_i << input_SpeedBias_i[6], input_SpeedBias_i[7], input_SpeedBias_i[8];

    p_j << input_Pose_j[0], input_Pose_j[1], input_Pose_j[2];
    q_j.coeffs() << input_Pose_j[3], input_Pose_j[4], input_Pose_j[5], input_Pose_j[6];
    v_j << input_SpeedBias_j[0], input_SpeedBias_j[1], input_SpeedBias_j[2];
    bias_acc_j << input_SpeedBias_j[3], input_SpeedBias_j[4], input_SpeedBias_j[5];
    bias_gyr_j << input_SpeedBias_j[6], input_SpeedBias_j[7], input_SpeedBias_j[8];
}

template<>
SimpleIMUFactor<float>::SimpleIMUFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    involved_in_marg(0),
    gravity(input_gravity.cast<float>()),
    pre_integration_j_sum_dt(float(input_preinteg_ptr->sum_dt)),
    pre_integration_j_linearized_bias_acc(input_preinteg_ptr->linearized_ba.cast<float>()),
    pre_integration_j_linearized_bias_gyr(input_preinteg_ptr->linearized_bg.cast<float>()),
    pre_integration_j_delta_p(input_preinteg_ptr->delta_p.cast<float>()),
    pre_integration_j_delta_q(input_preinteg_ptr->delta_q.cast<float>()),
    pre_integration_j_delta_v(input_preinteg_ptr->delta_v.cast<float>()),
    pre_integration_j_jacobian(input_preinteg_ptr->jacobian.cast<float>()),
    pre_integration_j_covariance(input_preinteg_ptr->covariance.cast<float>()),
    pose_i_is_fixed(input_pose_i_is_fixed),
    speed_bias_i_is_fixed(input_speed_bias_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    speed_bias_j_is_fixed(input_speed_bias_j_is_fixed)
{
    jacobian_p_bias_acc = pre_integration_j_jacobian.block<3, 3>(O_P, O_BA);
    jacobian_p_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_P, O_BG);
    jacobian_q_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_R, O_BG);
    jacobian_v_bias_acc = pre_integration_j_jacobian.block<3, 3>(O_V, O_BA);
    jacobian_v_bias_gyr = pre_integration_j_jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Matrix<float, 7, 1> input_Pose_tmp = input_Pose_i.cast<float>();
    Eigen::Matrix<float, 9, 1> input_SpeedBias_tmp = input_SpeedBias_i.cast<float>();

    p_i << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_i.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];
    v_i << input_SpeedBias_tmp[0], input_SpeedBias_tmp[1], input_SpeedBias_tmp[2];
    bias_acc_i << input_SpeedBias_tmp[3], input_SpeedBias_tmp[4], input_SpeedBias_tmp[5];
    bias_gyr_i << input_SpeedBias_tmp[6], input_SpeedBias_tmp[7], input_SpeedBias_tmp[8];

    input_Pose_tmp = input_Pose_j.cast<float>();
    input_SpeedBias_tmp = input_SpeedBias_j.cast<float>();

    p_j << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_j.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];
    v_j << input_SpeedBias_tmp[0], input_SpeedBias_tmp[1], input_SpeedBias_tmp[2];
    bias_acc_j << input_SpeedBias_tmp[3], input_SpeedBias_tmp[4], input_SpeedBias_tmp[5];
    bias_gyr_j << input_SpeedBias_tmp[6], input_SpeedBias_tmp[7], input_SpeedBias_tmp[8];
}

// ----------

template<>
SimpleProj2F2CFactor<double>::SimpleProj2F2CFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i),
    pts_j(input_pts_j),
    td_i(input_td_i),
    td_j(input_td_j),
    inv_depth(input_inv_depth),
    time_delay(input_time_delay),
    pose_i_is_fixed(input_pose_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    pose_ex_1_is_fixed(input_pose_ex_1_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    velocity_i << input_velocity_i[0], input_velocity_i[1], 0.0;
    velocity_j << input_velocity_j[0], input_velocity_j[1], 0.0;

    p_i << input_Pose_i[0], input_Pose_i[1], input_Pose_i[2];
    q_i.coeffs() << input_Pose_i[3], input_Pose_i[4], input_Pose_i[5], input_Pose_i[6];

    p_j << input_Pose_j[0], input_Pose_j[1], input_Pose_j[2];
    q_j.coeffs() << input_Pose_j[3], input_Pose_j[4], input_Pose_j[5], input_Pose_j[6];

    p_ex_0 << input_Ex_Pose_0[0], input_Ex_Pose_0[1], input_Ex_Pose_0[2];
    q_ex_0.coeffs() << input_Ex_Pose_0[3], input_Ex_Pose_0[4], input_Ex_Pose_0[5], input_Ex_Pose_0[6];

    p_ex_1 << input_Ex_Pose_1[0], input_Ex_Pose_1[1], input_Ex_Pose_1[2];
    q_ex_1.coeffs() << input_Ex_Pose_1[3], input_Ex_Pose_1[4], input_Ex_Pose_1[5], input_Ex_Pose_1[6];
}

template<>
SimpleProj2F2CFactor<float>::SimpleProj2F2CFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i.cast<float>()),
    pts_j(input_pts_j.cast<float>()),
    td_i(float(input_td_i)),
    td_j(float(input_td_j)),
    inv_depth(float(input_inv_depth)),
    time_delay(float(input_time_delay)),
    pose_i_is_fixed(input_pose_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    pose_ex_1_is_fixed(input_pose_ex_1_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    Eigen::Vector2f velocity_tmp = input_velocity_i.cast<float>();
    velocity_i << velocity_tmp[0], velocity_tmp[1], 0.0;
    velocity_tmp = input_velocity_j.cast<float>();
    velocity_j << velocity_tmp[0], velocity_tmp[1], 0.0;

    Eigen::Matrix<float, 7, 1> input_Pose_tmp = input_Pose_i.cast<float>();

    p_i << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_i.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Pose_j.cast<float>();

    p_j << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_j.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Ex_Pose_0.cast<float>();

    p_ex_0 << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_ex_0.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Ex_Pose_1.cast<float>();

    p_ex_1 << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_ex_1.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];
}

// ----------

template<>
SimpleProj2F1CFactor<double>::SimpleProj2F1CFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i),
    pts_j(input_pts_j),
    td_i(input_td_i),
    td_j(input_td_j),
    inv_depth(input_inv_depth),
    time_delay(input_time_delay),
    pose_i_is_fixed(input_pose_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    velocity_i << input_velocity_i[0], input_velocity_i[1], 0.0;
    velocity_j << input_velocity_j[0], input_velocity_j[1], 0.0;

    p_i << input_Pose_i[0], input_Pose_i[1], input_Pose_i[2];
    q_i.coeffs() << input_Pose_i[3], input_Pose_i[4], input_Pose_i[5], input_Pose_i[6];

    p_j << input_Pose_j[0], input_Pose_j[1], input_Pose_j[2];
    q_j.coeffs() << input_Pose_j[3], input_Pose_j[4], input_Pose_j[5], input_Pose_j[6];

    p_ex_0 << input_Ex_Pose_0[0], input_Ex_Pose_0[1], input_Ex_Pose_0[2];
    q_ex_0.coeffs() << input_Ex_Pose_0[3], input_Ex_Pose_0[4], input_Ex_Pose_0[5], input_Ex_Pose_0[6];
}

template<>
SimpleProj2F1CFactor<float>::SimpleProj2F1CFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i.cast<float>()),
    pts_j(input_pts_j.cast<float>()),
    td_i(float(input_td_i)),
    td_j(float(input_td_j)),
    inv_depth(float(input_inv_depth)),
    time_delay(float(input_time_delay)),
    pose_i_is_fixed(input_pose_i_is_fixed),
    pose_j_is_fixed(input_pose_j_is_fixed),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    Eigen::Vector2f velocity_tmp = input_velocity_i.cast<float>();
    velocity_i << velocity_tmp[0], velocity_tmp[1], 0.0;
    velocity_tmp = input_velocity_j.cast<float>();
    velocity_j << velocity_tmp[0], velocity_tmp[1], 0.0;

    Eigen::Matrix<float, 7, 1> input_Pose_tmp = input_Pose_i.cast<float>();

    p_i << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_i.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Pose_j.cast<float>();

    p_j << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_j.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Ex_Pose_0.cast<float>();

    p_ex_0 << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_ex_0.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];
}

// ----------

template<>
SimpleProj1F2CFactor<double>::SimpleProj1F2CFactor(
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
) 
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i),
    pts_j(input_pts_j),
    td_i(input_td_i),
    td_j(input_td_j),
    inv_depth(input_inv_depth),
    time_delay(input_time_delay),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    pose_ex_1_is_fixed(input_pose_ex_1_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    velocity_i << input_velocity_i[0], input_velocity_i[1], 0.0;
    velocity_j << input_velocity_j[0], input_velocity_j[1], 0.0;

    p_ex_0 << input_Ex_Pose_0[0], input_Ex_Pose_0[1], input_Ex_Pose_0[2];
    q_ex_0.coeffs() << input_Ex_Pose_0[3], input_Ex_Pose_0[4], input_Ex_Pose_0[5], input_Ex_Pose_0[6];

    p_ex_1 << input_Ex_Pose_1[0], input_Ex_Pose_1[1], input_Ex_Pose_1[2];
    q_ex_1.coeffs() << input_Ex_Pose_1[3], input_Ex_Pose_1[4], input_Ex_Pose_1[5], input_Ex_Pose_1[6];
}

template<>
SimpleProj1F2CFactor<float>::SimpleProj1F2CFactor(
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
)
  : idx_i(input_idx_i),
    idx_j(input_idx_j),
    feature_index(input_feature_index),
    feature_index_for_marg(-1),
    involved_in_marg(0),
    pts_i(input_pts_i.cast<float>()),
    pts_j(input_pts_j.cast<float>()),
    td_i(float(input_td_i)),
    td_j(float(input_td_j)),
    inv_depth(float(input_inv_depth)),
    time_delay(float(input_time_delay)),
    pose_ex_0_is_fixed(input_pose_ex_0_is_fixed),
    pose_ex_1_is_fixed(input_pose_ex_1_is_fixed),
    inv_depth_is_fixed(input_inv_depth_is_fixed),
    cur_td_is_fixed(input_cur_td_is_fixed)
{
    Eigen::Vector2f velocity_tmp = input_velocity_i.cast<float>();
    velocity_i << velocity_tmp[0], velocity_tmp[1], 0.0;
    velocity_tmp = input_velocity_j.cast<float>();
    velocity_j << velocity_tmp[0], velocity_tmp[1], 0.0;

    Eigen::Matrix<float, 7, 1> input_Pose_tmp = input_Ex_Pose_0.cast<float>();

    p_ex_0 << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_ex_0.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];

    input_Pose_tmp = input_Ex_Pose_1.cast<float>();

    p_ex_1 << input_Pose_tmp[0], input_Pose_tmp[1], input_Pose_tmp[2];
    q_ex_1.coeffs() << input_Pose_tmp[3], input_Pose_tmp[4], input_Pose_tmp[5], input_Pose_tmp[6];
}

// ----------

// instantiation for T = double

template class SimpleIMUFactor<double>;
template class SimpleProj2F2CFactor<double>;
template class SimpleProj2F1CFactor<double>;
template class SimpleProj1F2CFactor<double>;

// ----------

// instantiation for T = float

template class SimpleIMUFactor<float>;
template class SimpleProj2F2CFactor<float>;
template class SimpleProj2F1CFactor<float>;
template class SimpleProj1F2CFactor<float>;

} // namespace VINS_FUSION_CUDA_BA

