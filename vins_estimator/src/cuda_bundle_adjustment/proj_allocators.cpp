//
// Created by lmf on 23-7-20.
//

#include "proj_allocators.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void Proj2F2CFactorAllocator<T>::Alloc(const std::vector< SimpleProj2F2CFactor<T> >& factor_vec) {
    num_factors = factor_vec.size();

    imu_idx_i.resize(num_factors, 1); imu_idx_i.setZero();
    imu_idx_j.resize(num_factors, 1); imu_idx_j.setZero();

    inv_depth_idx.resize(num_factors, 1); inv_depth_idx.setZero();
    inv_depth_idx_for_marg.resize(num_factors, 1); inv_depth_idx_for_marg.setZero();
    involved_in_marg.resize(num_factors, 1); involved_in_marg.setZero();

    pose_i_is_fixed.resize(num_factors, 1); pose_i_is_fixed.setZero();
    pose_j_is_fixed.resize(num_factors, 1); pose_j_is_fixed.setZero();
    pose_ex_0_is_fixed.resize(num_factors, 1); pose_ex_0_is_fixed.setZero();
    pose_ex_1_is_fixed.resize(num_factors, 1); pose_ex_1_is_fixed.setZero();
    inv_depth_is_fixed.resize(num_factors, 1); inv_depth_is_fixed.setZero();
    cur_td_is_fixed.resize(num_factors, 1); cur_td_is_fixed.setZero();

    td_i.resize(num_factors, 1); td_i.setZero();
    td_j.resize(num_factors, 1); td_j.setZero();

    inv_depth.resize(num_factors, 1); inv_depth.setZero();
    time_delay.resize(num_factors, 1); time_delay.setZero();

    // ----------

    pts_i_v1.resize(3, num_factors); pts_i_v1.setZero();
    pts_j_v1.resize(3, num_factors); pts_j_v1.setZero();

    velocity_i_v1.resize(3, num_factors); velocity_i_v1.setZero();
    velocity_j_v1.resize(3, num_factors); velocity_j_v1.setZero();

    p_i_v1.resize(3, num_factors); p_i_v1.setZero();
    q_i_v1.resize(4, num_factors); q_i_v1.setZero();

    p_j_v1.resize(3, num_factors); p_j_v1.setZero();
    q_j_v1.resize(4, num_factors); q_j_v1.setZero();

    p_ex_0_v1.resize(3, num_factors); p_ex_0_v1.setZero();
    q_ex_0_v1.resize(4, num_factors); q_ex_0_v1.setZero();

    p_ex_1_v1.resize(3, num_factors); p_ex_1_v1.setZero();
    q_ex_1_v1.resize(4, num_factors); q_ex_1_v1.setZero();

    #pragma omp parallel for num_threads(numberOfCores)
    for(int i = 0; i < factor_vec.size(); i++) {
        const auto &factor = factor_vec[i];

        imu_idx_i(i) = factor.idx_i;
        imu_idx_j(i) = factor.idx_j;

        inv_depth_idx(i) = factor.feature_index;
        inv_depth_idx_for_marg(i) = factor.feature_index_for_marg;
        involved_in_marg(i) = factor.involved_in_marg;

        pose_i_is_fixed(i) = factor.pose_i_is_fixed ? 1 : 0;
        pose_j_is_fixed(i) = factor.pose_j_is_fixed ? 1 : 0;
        pose_ex_0_is_fixed(i) = factor.pose_ex_0_is_fixed ? 1 : 0;
        pose_ex_1_is_fixed(i) = factor.pose_ex_1_is_fixed ? 1 : 0;
        inv_depth_is_fixed(i) = factor.inv_depth_is_fixed ? 1 : 0;
        cur_td_is_fixed(i) = factor.cur_td_is_fixed ? 1 : 0;

        td_i(i) = factor.td_i;
        td_j(i) = factor.td_j;

        inv_depth(i) = factor.inv_depth;
        time_delay(i) = factor.time_delay;

        // ----------

        pts_i_v1.col(i) = factor.pts_i;
        pts_j_v1.col(i) = factor.pts_j;

        velocity_i_v1.col(i) = factor.velocity_i;
        velocity_j_v1.col(i) = factor.velocity_j;

        p_i_v1.col(i) = factor.p_i;
        q_i_v1.col(i) = factor.q_i.coeffs();

        p_j_v1.col(i) = factor.p_j;
        q_j_v1.col(i) = factor.q_j.coeffs();

        p_ex_0_v1.col(i) = factor.p_ex_0;
        q_ex_0_v1.col(i) = factor.q_ex_0.coeffs();

        p_ex_1_v1.col(i) = factor.p_ex_1;
        q_ex_1_v1.col(i) = factor.q_ex_1.coeffs();
    }

    pts_i_v2 = pts_i_v1;
    pts_j_v2 = pts_j_v1;

    velocity_i_v2 = velocity_i_v1;
    velocity_j_v2 = velocity_j_v1;

    p_i_v2 = p_i_v1;
    q_i_v2 = q_i_v1;

    p_j_v2 = p_j_v1;
    q_j_v2 = q_j_v1;

    p_ex_0_v2 = p_ex_0_v1;
    q_ex_0_v2 = q_ex_0_v1;

    p_ex_1_v2 = p_ex_1_v1;
    q_ex_1_v2 = q_ex_1_v1;
}

// ----------

template<typename T>
void Proj2F1CFactorAllocator<T>::Alloc(const std::vector< SimpleProj2F1CFactor<T> >& factor_vec) {
    num_factors = factor_vec.size();

    imu_idx_i.resize(num_factors, 1); imu_idx_i.setZero();
    imu_idx_j.resize(num_factors, 1); imu_idx_j.setZero();

    inv_depth_idx.resize(num_factors, 1); inv_depth_idx.setZero();
    inv_depth_idx_for_marg.resize(num_factors, 1); inv_depth_idx_for_marg.setZero();
    involved_in_marg.resize(num_factors, 1); involved_in_marg.setZero();

    pose_i_is_fixed.resize(num_factors, 1); pose_i_is_fixed.setZero();
    pose_j_is_fixed.resize(num_factors, 1); pose_j_is_fixed.setZero();
    pose_ex_0_is_fixed.resize(num_factors, 1); pose_ex_0_is_fixed.setZero();
    inv_depth_is_fixed.resize(num_factors, 1); inv_depth_is_fixed.setZero();
    cur_td_is_fixed.resize(num_factors, 1); cur_td_is_fixed.setZero();

    td_i.resize(num_factors, 1); td_i.setZero();
    td_j.resize(num_factors, 1); td_j.setZero();

    inv_depth.resize(num_factors, 1); inv_depth.setZero();
    time_delay.resize(num_factors, 1); time_delay.setZero();

    // ----------

    pts_i_v1.resize(3, num_factors); pts_i_v1.setZero();
    pts_j_v1.resize(3, num_factors); pts_j_v1.setZero();

    velocity_i_v1.resize(3, num_factors); velocity_i_v1.setZero();
    velocity_j_v1.resize(3, num_factors); velocity_j_v1.setZero();

    p_i_v1.resize(3, num_factors); p_i_v1.setZero();
    q_i_v1.resize(4, num_factors); q_i_v1.setZero();

    p_j_v1.resize(3, num_factors); p_j_v1.setZero();
    q_j_v1.resize(4, num_factors); q_j_v1.setZero();

    p_ex_0_v1.resize(3, num_factors); p_ex_0_v1.setZero();
    q_ex_0_v1.resize(4, num_factors); q_ex_0_v1.setZero();

    #pragma omp parallel for num_threads(numberOfCores)
    for(int i = 0; i < factor_vec.size(); i++) {
        const auto &factor = factor_vec[i];

        imu_idx_i(i) = factor.idx_i;
        imu_idx_j(i) = factor.idx_j;

        inv_depth_idx(i) = factor.feature_index;
        inv_depth_idx_for_marg(i) = factor.feature_index_for_marg;
        involved_in_marg(i) = factor.involved_in_marg;

        pose_i_is_fixed(i) = factor.pose_i_is_fixed ? 1 : 0;
        pose_j_is_fixed(i) = factor.pose_j_is_fixed ? 1 : 0;
        pose_ex_0_is_fixed(i) = factor.pose_ex_0_is_fixed ? 1 : 0;
        inv_depth_is_fixed(i) = factor.inv_depth_is_fixed ? 1 : 0;
        cur_td_is_fixed(i) = factor.cur_td_is_fixed ? 1 : 0;

        td_i(i) = factor.td_i;
        td_j(i) = factor.td_j;

        inv_depth(i) = factor.inv_depth;
        time_delay(i) = factor.time_delay;

        // ----------

        pts_i_v1.col(i) = factor.pts_i;
        pts_j_v1.col(i) = factor.pts_j;

        velocity_i_v1.col(i) = factor.velocity_i;
        velocity_j_v1.col(i) = factor.velocity_j;

        p_i_v1.col(i) = factor.p_i;
        q_i_v1.col(i) = factor.q_i.coeffs();

        p_j_v1.col(i) = factor.p_j;
        q_j_v1.col(i) = factor.q_j.coeffs();

        p_ex_0_v1.col(i) = factor.p_ex_0;
        q_ex_0_v1.col(i) = factor.q_ex_0.coeffs();
    }

    pts_i_v2 = pts_i_v1;
    pts_j_v2 = pts_j_v1;

    velocity_i_v2 = velocity_i_v1;
    velocity_j_v2 = velocity_j_v1;

    p_i_v2 = p_i_v1;
    q_i_v2 = q_i_v1;

    p_j_v2 = p_j_v1;
    q_j_v2 = q_j_v1;

    p_ex_0_v2 = p_ex_0_v1;
    q_ex_0_v2 = q_ex_0_v1;
}

// ----------

template<typename T>
void Proj1F2CFactorAllocator<T>::Alloc(const std::vector< SimpleProj1F2CFactor<T> >& factor_vec) {
    num_factors = factor_vec.size();

    imu_idx_i.resize(num_factors, 1); imu_idx_i.setZero();
    imu_idx_j.resize(num_factors, 1); imu_idx_j.setZero();

    inv_depth_idx.resize(num_factors, 1); inv_depth_idx.setZero();
    inv_depth_idx_for_marg.resize(num_factors, 1); inv_depth_idx_for_marg.setZero();
    involved_in_marg.resize(num_factors, 1); involved_in_marg.setZero();

    pose_ex_0_is_fixed.resize(num_factors, 1); pose_ex_0_is_fixed.setZero();
    pose_ex_1_is_fixed.resize(num_factors, 1); pose_ex_1_is_fixed.setZero();
    inv_depth_is_fixed.resize(num_factors, 1); inv_depth_is_fixed.setZero();
    cur_td_is_fixed.resize(num_factors, 1); cur_td_is_fixed.setZero();

    td_i.resize(num_factors, 1); td_i.setZero();
    td_j.resize(num_factors, 1); td_j.setZero();

    inv_depth.resize(num_factors, 1); inv_depth.setZero();
    time_delay.resize(num_factors, 1); time_delay.setZero();

    // ----------

    pts_i_v1.resize(3, num_factors); pts_i_v1.setZero();
    pts_j_v1.resize(3, num_factors); pts_j_v1.setZero();

    velocity_i_v1.resize(3, num_factors); velocity_i_v1.setZero();
    velocity_j_v1.resize(3, num_factors); velocity_j_v1.setZero();

    p_ex_0_v1.resize(3, num_factors); p_ex_0_v1.setZero();
    q_ex_0_v1.resize(4, num_factors); q_ex_0_v1.setZero();

    p_ex_1_v1.resize(3, num_factors); p_ex_1_v1.setZero();
    q_ex_1_v1.resize(4, num_factors); q_ex_1_v1.setZero();

    #pragma omp parallel for num_threads(numberOfCores)
    for(int i = 0; i < factor_vec.size(); i++) {
        const auto &factor = factor_vec[i];

        imu_idx_i(i) = factor.idx_i;
        imu_idx_j(i) = factor.idx_j;

        inv_depth_idx(i) = factor.feature_index;
        inv_depth_idx_for_marg(i) = factor.feature_index_for_marg;
        involved_in_marg(i) = factor.involved_in_marg;

        pose_ex_0_is_fixed(i) = factor.pose_ex_0_is_fixed ? 1 : 0;
        pose_ex_1_is_fixed(i) = factor.pose_ex_1_is_fixed ? 1 : 0;
        inv_depth_is_fixed(i) = factor.inv_depth_is_fixed ? 1 : 0;
        cur_td_is_fixed(i) = factor.cur_td_is_fixed ? 1 : 0;

        td_i(i) = factor.td_i;
        td_j(i) = factor.td_j;

        inv_depth(i) = factor.inv_depth;
        time_delay(i) = factor.time_delay;

        // ----------

        pts_i_v1.col(i) = factor.pts_i;
        pts_j_v1.col(i) = factor.pts_j;

        velocity_i_v1.col(i) = factor.velocity_i;
        velocity_j_v1.col(i) = factor.velocity_j;

        p_ex_0_v1.col(i) = factor.p_ex_0;
        q_ex_0_v1.col(i) = factor.q_ex_0.coeffs();

        p_ex_1_v1.col(i) = factor.p_ex_1;
        q_ex_1_v1.col(i) = factor.q_ex_1.coeffs();
    }

    pts_i_v2 = pts_i_v1;
    pts_j_v2 = pts_j_v1;

    velocity_i_v2 = velocity_i_v1;
    velocity_j_v2 = velocity_j_v1;

    p_ex_0_v2 = p_ex_0_v1;
    q_ex_0_v2 = q_ex_0_v1;

    p_ex_1_v2 = p_ex_1_v1;
    q_ex_1_v2 = q_ex_1_v1;
}

// ----------

// instantiation for T = double

template class Proj2F2CFactorAllocator<double>;
template class Proj2F1CFactorAllocator<double>;
template class Proj1F2CFactorAllocator<double>;

// ----------

// instantiation for T = float

template class Proj2F2CFactorAllocator<float>;
template class Proj2F1CFactorAllocator<float>;
template class Proj1F2CFactorAllocator<float>;

} // namespace VINS_FUSION_CUDA_BA
