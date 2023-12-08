//
// Created by lmf on 23-7-20.
//

#ifndef CUDA_BA_PROJ_ALLOCATORS_H
#define CUDA_BA_PROJ_ALLOCATORS_H

#include "factors.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class Proj2F2CFactorAllocator {

public :
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_i;
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_j;

    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx;
    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx_for_marg;
    Eigen::Matrix<char, Eigen::Dynamic, 1> involved_in_marg;

    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_i_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_j_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_ex_0_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_ex_1_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> inv_depth_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> cur_td_is_fixed;

    Eigen::Matrix<T, Eigen::Dynamic, 1> td_i;
    Eigen::Matrix<T, Eigen::Dynamic, 1> td_j;
    Eigen::Matrix<T, Eigen::Dynamic, 1> inv_depth;
    Eigen::Matrix<T, Eigen::Dynamic, 1> time_delay;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_i_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_ex_0_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_ex_0_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_ex_1_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_ex_1_v1;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_i_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_ex_0_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_ex_0_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_ex_1_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_ex_1_v2;

public :
    Proj2F2CFactorAllocator() = default;

public :
    size_t num_factors = 0;

public :
    void Alloc(const std::vector< SimpleProj2F2CFactor<T> >& factor_vec);
};

// ----------

template<typename T>
class Proj2F1CFactorAllocator {

public :
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_i;
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_j;

    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx;
    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx_for_marg;
    Eigen::Matrix<char, Eigen::Dynamic, 1> involved_in_marg;

    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_i_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_j_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_ex_0_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> inv_depth_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> cur_td_is_fixed;

    Eigen::Matrix<T, Eigen::Dynamic, 1> td_i;
    Eigen::Matrix<T, Eigen::Dynamic, 1> td_j;
    Eigen::Matrix<T, Eigen::Dynamic, 1> inv_depth;
    Eigen::Matrix<T, Eigen::Dynamic, 1> time_delay;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_i_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_j_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_ex_0_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_ex_0_v1;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_i_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_j_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_ex_0_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_ex_0_v2;

public :
    Proj2F1CFactorAllocator() = default;

public :
    size_t num_factors = 0;

public :
    void Alloc(const std::vector< SimpleProj2F1CFactor<T> >& factor_vec);
};

// ----------

template<typename T>
class Proj1F2CFactorAllocator {

public :
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_i;
    Eigen::Matrix<int, Eigen::Dynamic, 1> imu_idx_j;

    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx;
    Eigen::Matrix<int, Eigen::Dynamic, 1> inv_depth_idx_for_marg;
    Eigen::Matrix<char, Eigen::Dynamic, 1> involved_in_marg;

    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_ex_0_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> pose_ex_1_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> inv_depth_is_fixed;
    Eigen::Matrix<char, Eigen::Dynamic, 1> cur_td_is_fixed;

    Eigen::Matrix<T, Eigen::Dynamic, 1> td_i;
    Eigen::Matrix<T, Eigen::Dynamic, 1> td_j;
    Eigen::Matrix<T, Eigen::Dynamic, 1> inv_depth;
    Eigen::Matrix<T, Eigen::Dynamic, 1> time_delay;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> pts_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_i_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> velocity_j_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_ex_0_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_ex_0_v1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> p_ex_1_v1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q_ex_1_v1;

    // ----------

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_i_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> velocity_j_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_ex_0_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_ex_0_v2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_ex_1_v2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_ex_1_v2;

public :
    Proj1F2CFactorAllocator() = default;

public :
    size_t num_factors = 0;

public :
    void Alloc(const std::vector< SimpleProj1F2CFactor<T> >& factor_vec);
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_PROJ_ALLOCATORS_H
