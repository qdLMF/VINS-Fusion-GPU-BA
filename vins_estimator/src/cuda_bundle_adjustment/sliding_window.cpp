//
// Created by lmf on 23-7-22.
//

#include "sliding_window.h"

#include "cuda_kernel_funcs/cublas_funcs.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
SlidingWindow<T>::SlidingWindow(int _num_key_frames, bool _decompose_prior) {
    assert(_num_key_frames >= 2);
    num_key_frames = _num_key_frames;
    max_num_imu_factors = _num_key_frames - 1;
    decompose_prior = _decompose_prior;

    have_prior = false;

    Init();
}

template<typename T>
void SlidingWindow<T>::Init() {
    host_states.resize(16 * num_key_frames, 1);
    host_states_backup.resize(16 * num_key_frames, 1);
    host_states_init.resize(16 * num_key_frames, 1);

    shapes.ComputeMaxSizes(max_num_imu_factors, max_num_world_points);

    assert(AllocForStates());
    assert(AllocForDeviceBigHessiansAndRHS());

    InitCUDAHandles();
}

template<typename T>
void SlidingWindow<T>::ComputeShapes() {
    prev_keyframe_idx_in_marg = curr_keyframe_idx_in_marg;
    curr_keyframe_idx_in_marg.clear();

    for (auto &imu_factor: imu_factor_vec) {
        if (imu_factor.idx_i == marg_keyframe_idx || imu_factor.idx_j == marg_keyframe_idx) {
            imu_factor.involved_in_marg = 1;
            if (curr_keyframe_idx_in_marg.find(imu_factor.idx_i) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(imu_factor.idx_i);
            }
            if (curr_keyframe_idx_in_marg.find(imu_factor.idx_j) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(imu_factor.idx_j);
            }
        }
    }

    std::map<int, int> feature_index_old_new;
    for (const auto &proj_factor: proj_2f1c_factor_vec) {
        if (feature_index_old_new.find(proj_factor.feature_index) == feature_index_old_new.end()) {
            feature_index_old_new.emplace(proj_factor.feature_index, feature_index_old_new.size());
        }
    }
    for (const auto &proj_factor: proj_2f2c_factor_vec) {
        if (feature_index_old_new.find(proj_factor.feature_index) == feature_index_old_new.end()) {
            feature_index_old_new.emplace(proj_factor.feature_index, feature_index_old_new.size());
        }
    }
    for (const auto &proj_factor: proj_1f2c_factor_vec) {
        if (feature_index_old_new.find(proj_factor.feature_index) == feature_index_old_new.end()) {
            feature_index_old_new.emplace(proj_factor.feature_index, feature_index_old_new.size());
        }
    }
    int num_world_points = feature_index_old_new.size();

    std::map<int, int> feature_index_old_new_for_marg;
    for (auto &proj_factor: proj_2f1c_factor_vec) {
        if (proj_factor.idx_i == marg_keyframe_idx) {
            if (feature_index_old_new_for_marg.find(proj_factor.feature_index) ==
                feature_index_old_new_for_marg.end()) {
                feature_index_old_new_for_marg.emplace(proj_factor.feature_index,
                                                       feature_index_old_new_for_marg.size());
            }
            proj_factor.feature_index_for_marg = feature_index_old_new_for_marg[proj_factor.feature_index];
            proj_factor.involved_in_marg = 1;
            if (curr_keyframe_idx_in_marg.find(proj_factor.idx_i) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(proj_factor.idx_i);
            }
            if (curr_keyframe_idx_in_marg.find(proj_factor.idx_j) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(proj_factor.idx_j);
            }
        }
    }
    for (auto &proj_factor: proj_2f2c_factor_vec) {
        if (proj_factor.idx_i == marg_keyframe_idx) {
            if (feature_index_old_new_for_marg.find(proj_factor.feature_index) ==
                feature_index_old_new_for_marg.end()) {
                feature_index_old_new_for_marg.emplace(proj_factor.feature_index,
                                                       feature_index_old_new_for_marg.size());
            }
            proj_factor.feature_index_for_marg = feature_index_old_new_for_marg[proj_factor.feature_index];
            proj_factor.involved_in_marg = 1;
            if (curr_keyframe_idx_in_marg.find(proj_factor.idx_i) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(proj_factor.idx_i);
            }
            if (curr_keyframe_idx_in_marg.find(proj_factor.idx_j) == curr_keyframe_idx_in_marg.end()) {
                curr_keyframe_idx_in_marg.insert(proj_factor.idx_j);
            }
        }
    }
    for (auto &proj_factor: proj_1f2c_factor_vec) {
        if (proj_factor.idx_i == marg_keyframe_idx) {
            if (feature_index_old_new_for_marg.find(proj_factor.feature_index) ==
                feature_index_old_new_for_marg.end()) {
                feature_index_old_new_for_marg.emplace(proj_factor.feature_index,
                                                       feature_index_old_new_for_marg.size());
            }
            proj_factor.feature_index_for_marg = feature_index_old_new_for_marg[proj_factor.feature_index];
            proj_factor.involved_in_marg = 1;
        }
    }
    int num_world_points_for_marg = feature_index_old_new_for_marg.size();

    shapes.ComputeActualSizes(
        imu_factor_vec.size(),
        proj_2f1c_factor_vec.size(),
        proj_2f2c_factor_vec.size(),
        proj_1f2c_factor_vec.size(),
        num_world_points,
        num_world_points_for_marg
    );
}

// ----------

template<typename T>
bool SlidingWindow<T>::AllocForStates() {
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_0" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_1, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_1" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_states, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_states" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_cur_td, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_cur_td" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_0_init, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_0_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_1_init, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_1_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_states_init, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_states_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_cur_td_init, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_cur_td_init" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_0_backup, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_0_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_ex_para_1_backup, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_ex_para_1_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_states_backup, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_states_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_cur_td_backup, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_cur_td_backup" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_ex_para_0, shapes.num_elem_delta_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_ex_para_0" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_ex_para_1, shapes.num_elem_delta_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_ex_para_1" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_states, shapes.num_elem_delta_states * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_states" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_cur_td, shapes.num_elem_delta_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_cur_td" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_lambda, 1 * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_lambda" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_inv_depth, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_inv_depth" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_inv_depth_init, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_inv_depth_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_inv_depth_backup, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_inv_depth_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_inv_depth, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_inv_depth" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
bool SlidingWindow<T>::AllocForDeviceBigHessiansAndRHS() {
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hmm_diag, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hmm_diag" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpm, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpm" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hmp, sizeof(T) * shapes.max_num_elem_Hmp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hmp" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bmm, sizeof(T) * shapes.max_num_elem_Bmm) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bmm" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hmm_diag_for_marg, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hmm_diag_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpm_for_marg, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpm_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hmp_for_marg, sizeof(T) * shapes.max_num_elem_Hmp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hmp_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bmm_for_marg, sizeof(T) * shapes.max_num_elem_Bmm) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bmm_for_marg" << std::endl;
        return false;
    }

    // ----------

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpp, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpp" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bpp, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bpp" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpp_with_lambda, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpp_with_lambda" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hmm_diag_with_lambda, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hmm_diag_with_lambda" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_HpmHmmInv, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_HpmHmmInv" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpp_schur, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpp_schur" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bpp_schur, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bpp_schur" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_delta_Xpp, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_delta_Xpp" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hpp_for_marg, sizeof(T) * shapes.num_rows_Hpp_for_marg * shapes.num_cols_Hpp_for_marg) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hpp_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bpp_for_marg, sizeof(T) * shapes.num_elem_Bpp_for_marg) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bpp_for_marg" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_H11, sizeof(T) * shapes.num_rows_H11 * shapes.num_cols_H11) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_H11" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_H12, sizeof(T) * shapes.num_rows_H12 * shapes.num_cols_H12) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_H12" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_H21, sizeof(T) * shapes.num_rows_H21 * shapes.num_cols_H21) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_H21" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_H22, sizeof(T) * shapes.num_rows_H22 * shapes.num_cols_H22) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_H22" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_B11, sizeof(T) * shapes.num_elem_B11) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_B11" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_B22, sizeof(T) * shapes.num_elem_B22) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_B22" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hprior, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Bprior, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Bprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Jprior, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Jprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Eprior, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Eprior" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hprior_eigenvec, sizeof(T) * shapes.num_rows_Hprior_eigenvec * shapes.num_cols_Hprior_eigenvec) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hprior_eigenvec" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMalloc((void**)&dev_ptr_Hprior_eigenval, sizeof(T) * shapes.num_elem_Hprior_eigenval) ) {
        std::cout << "CUDA Malloc Error : dev_ptr_Hprior_eigenval" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
bool SlidingWindow<T>::ClearForStates() {
    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_0, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_0" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_1, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_1" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_states, 0, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_states" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_cur_td, 0, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_cur_td" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_0_init, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_0_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_1_init, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_1_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_states_init, 0, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_states_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_cur_td_init, 0, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_cur_td_init" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_0_backup, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_0_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_ex_para_1_backup, 0, shapes.num_elem_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_ex_para_1_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_states_backup, 0, shapes.num_elem_states * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_states_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_cur_td_backup, 0, shapes.num_elem_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_cur_td_backup" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_delta_ex_para_0, 0, shapes.num_elem_delta_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_ex_para_0" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_delta_ex_para_1, 0, shapes.num_elem_delta_ex_para * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_ex_para_1" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_delta_states, 0, shapes.num_elem_delta_states * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_states" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_delta_cur_td, 0, shapes.num_elem_delta_cur_td * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_cur_td" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_lambda, 0, 1 * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_lambda" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_inv_depth, 0, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_inv_depth" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_inv_depth_init, 0, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_inv_depth_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_inv_depth_backup, 0, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_inv_depth_backup" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_delta_inv_depth, 0, shapes.max_num_elem_inv_depth * sizeof(T)) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_inv_depth" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
bool SlidingWindow<T>::ClearForDeviceBigHessiansAndRHS() {
    if( cudaSuccess != cudaMemset(dev_ptr_Hmm_diag, 0, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hmm_diag" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hpm, 0, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpm" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hmp, 0, sizeof(T) * shapes.max_num_elem_Hmp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hmp" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bmm, 0, sizeof(T) * shapes.max_num_elem_Bmm) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bmm" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_Hmm_diag_for_marg, 0, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hmm_diag_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hpm_for_marg, 0, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpm_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hmp_for_marg, 0, sizeof(T) * shapes.max_num_elem_Hmp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hmp_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bmm_for_marg, 0, sizeof(T) * shapes.max_num_elem_Bmm) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bmm_for_marg" << std::endl;
        return false;
    }

    // ----------

    if( cudaSuccess != cudaMemset(dev_ptr_Hpp, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpp" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bpp, 0, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bpp" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_Hpp_with_lambda, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpp_with_lambda" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hmm_diag_with_lambda, 0, sizeof(T) * shapes.max_num_elem_Hmm_diag) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hmm_diag_with_lambda" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_HpmHmmInv, 0, sizeof(T) * shapes.max_num_elem_Hpm) ) {
        std::cout << "CUDA Memset Error : dev_ptr_HpmHmmInv" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hpp_schur, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpp_schur" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bpp_schur, 0, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bpp_schur" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_delta_Xpp, 0, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_delta_Xpp" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_Hpp_for_marg, 0, sizeof(T) * shapes.num_rows_Hpp_for_marg * shapes.num_cols_Hpp_for_marg) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hpp_for_marg" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bpp_for_marg, 0, sizeof(T) * shapes.num_elem_Bpp_for_marg) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bpp_for_marg" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_H11, 0, sizeof(T) * shapes.num_rows_H11 * shapes.num_cols_H11) ) {
        std::cout << "CUDA Memset Error : dev_ptr_H11" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_H12, 0, sizeof(T) * shapes.num_rows_H12 * shapes.num_cols_H12) ) {
        std::cout << "CUDA Memset Error : dev_ptr_H12" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_H21, 0, sizeof(T) * shapes.num_rows_H21 * shapes.num_cols_H21) ) {
        std::cout << "CUDA Memset Error : dev_ptr_H21" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_H22, 0, sizeof(T) * shapes.num_rows_H22 * shapes.num_cols_H22) ) {
        std::cout << "CUDA Memset Error : dev_ptr_H22" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_B11, 0, sizeof(T) * shapes.num_elem_B11) ) {
        std::cout << "CUDA Memset Error : dev_ptr_B11" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_B22, 0, sizeof(T) * shapes.num_elem_B22) ) {
        std::cout << "CUDA Memset Error : dev_ptr_B22" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_Hprior, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Bprior, 0, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Bprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Jprior, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Jprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Eprior, 0, sizeof(T) * shapes.num_elem_Bpp) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Eprior" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemset(dev_ptr_Hprior_eigenvec, 0, sizeof(T) * shapes.num_rows_Hprior_eigenvec * shapes.num_cols_Hprior_eigenvec) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hprior_eigenvec" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemset(dev_ptr_Hprior_eigenval, 0, sizeof(T) * shapes.num_elem_Hprior_eigenval) ) {
        std::cout << "CUDA Memset Error : dev_ptr_Hprior_eigenval" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
void SlidingWindow<T>::MemcpyForCurrentFrame() {
    ComputeShapes();

    imu_allocator.Alloc(imu_factor_vec);
    proj_2f1c_allocator.Alloc(proj_2f1c_factor_vec);
    proj_2f2c_allocator.Alloc(proj_2f2c_factor_vec);
    proj_1f2c_allocator.Alloc(proj_1f2c_factor_vec);
    assert(imu_gpu_mem_pool.Alloc(imu_allocator));
    assert(proj_2f1c_gpu_mem_pool.Alloc(proj_2f1c_allocator));
    assert(proj_2f2c_gpu_mem_pool.Alloc(proj_2f2c_allocator));
    assert(proj_1f2c_gpu_mem_pool.Alloc(proj_1f2c_allocator));

    assert(MemcpyForStates());
    assert(MemcpyForDeviceBigHessiansAndRHS());

    lambda.num_key_frames  = shapes.num_key_frames;
    lambda.num_imu_factors = shapes.num_imu_factors;
    lambda.num_proj_2f1c_factors = shapes.num_proj_2f1c_factors;
    lambda.num_proj_2f2c_factors = shapes.num_proj_2f2c_factors;
    lambda.num_proj_1f2c_factors = shapes.num_proj_1f2c_factors;
    lambda.num_world_points = shapes.num_world_points;
    lambda.imu_dev_ptr = &(imu_gpu_mem_pool.dev_ptr_set);
    lambda.proj_2f1c_dev_ptr = &(proj_2f1c_gpu_mem_pool.dev_ptr_set);
    lambda.proj_2f2c_dev_ptr = &(proj_2f2c_gpu_mem_pool.dev_ptr_set);
    lambda.proj_1f2c_dev_ptr = &(proj_1f2c_gpu_mem_pool.dev_ptr_set);
    lambda.dev_ptr_lambda = dev_ptr_lambda;
    lambda.dev_ptr_Hpp = dev_ptr_Hpp;
    lambda.dev_ptr_Hmm_diag = dev_ptr_Hmm_diag;
    lambda.dev_ptr_Hpp_with_lambda = dev_ptr_Hpp_with_lambda;
    lambda.dev_ptr_Hmm_diag_with_lambda = dev_ptr_Hmm_diag_with_lambda;
    lambda.dev_ptr_delta_ex_para_0 = dev_ptr_delta_ex_para_0;
    lambda.dev_ptr_delta_ex_para_1 = dev_ptr_delta_ex_para_1;
    lambda.dev_ptr_delta_states = dev_ptr_delta_states;
    lambda.dev_ptr_delta_inv_depth = dev_ptr_delta_inv_depth;
    lambda.dev_ptr_delta_cur_td = dev_ptr_delta_cur_td;
    lambda.dev_ptr_Bpp = dev_ptr_Bpp;
    lambda.dev_ptr_Bmm = dev_ptr_Bmm;
    lambda.dev_ptr_Bprior = dev_ptr_Bprior;
    lambda.dev_ptr_err_prior = dev_ptr_Eprior;
    lambda.shapes = &shapes;

    P_init_vec.resize(num_key_frames);
    Q_init_vec.resize(num_key_frames);
    R_init_vec.resize(num_key_frames);
    V_init_vec.resize(num_key_frames);
    BiasAcc_init_vec.resize(num_key_frames);
    BiasGyr_init_vec.resize(num_key_frames);
    host_states_init.setZero();
    cudaMemcpy(host_states_init.data(), dev_ptr_states_init, sizeof(T) * num_key_frames * 16, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_key_frames; i++) {
        P_init_vec[i] = host_states_init.middleRows(i * 16 , 3);
        Q_init_vec[i] = Eigen::Quaternion<T>{ host_states_init(i * 16 + 3 + 3),     // real part goes first
                                              host_states_init(i * 16 + 3 + 0),
                                              host_states_init(i * 16 + 3 + 1),
                                              host_states_init(i * 16 + 3 + 2) }.normalized();
        R_init_vec[i] = Q_init_vec[i].toRotationMatrix();
        V_init_vec[i] = host_states_init.middleRows(i * 16 + 7 , 3);
        BiasAcc_init_vec[i] = host_states_init.middleRows(i * 16 + 10 , 3);
        BiasGyr_init_vec[i] = host_states_init.middleRows(i * 16 + 13 , 3);
    }
    
};

template<typename T>
bool SlidingWindow<T>::MemcpyForStates() {
    if( cudaSuccess != cudaMemcpy(dev_ptr_states, host_states.data(), sizeof(T) * num_key_frames * 16, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_states" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_ex_para_0, host_ex_para_0.data(), sizeof(T) * 7, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_ex_para_0" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_ex_para_1, host_ex_para_1.data(), sizeof(T) * 7, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_ex_para_1" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_cur_td, host_cur_td.data(), shapes.num_elem_cur_td * sizeof(T), cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_cur_td" << std::endl;
        return false;
    }

    if( cudaSuccess != cudaMemcpy(dev_ptr_states_init, host_states.data(), sizeof(T) * num_key_frames * 16, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_states_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_ex_para_0_init, host_ex_para_0.data(), sizeof(T) * 7, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_ex_para_0_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_ex_para_1_init, host_ex_para_1.data(), sizeof(T) * 7, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_ex_para_1_init" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_cur_td_init, host_cur_td.data(), shapes.num_elem_cur_td * sizeof(T), cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_cur_td" << std::endl;
        return false;
    }

    // ----------

    host_inv_depth.resize(shapes.num_elem_inv_depth); host_inv_depth.setZero();
    for(const auto& proj_factor : proj_2f1c_factor_vec) {
        host_inv_depth(proj_factor.feature_index) = proj_factor.inv_depth;
    }
   
    if( cudaSuccess != cudaMemcpy(dev_ptr_inv_depth, host_inv_depth.data(), shapes.num_elem_inv_depth * sizeof(T), cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_inv_depth" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_inv_depth_init, host_inv_depth.data(), shapes.num_elem_inv_depth * sizeof(T), cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_inv_depth_init" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
void SlidingWindow<T>::ClearForCurrentFrame() {
    assert(ClearForStates());
    assert(ClearForDeviceBigHessiansAndRHS());
}

// ----------

template<typename T>
bool SlidingWindow<T>::MemcpyForDeviceBigHessiansAndRHS() {
    if( cudaSuccess != cudaMemcpy(dev_ptr_Hprior, host_Hprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_Hprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_Bprior, host_Bprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_Bprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_Eprior, host_Eprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_Eprior" << std::endl;
        return false;
    }
    if( cudaSuccess != cudaMemcpy(dev_ptr_Jprior, host_Jprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice) ) {
        std::cout << "CUDA Memcpy Error : dev_ptr_Jprior" << std::endl;
        return false;
    }

    return true;
}

// ----------

template<typename T>
void SlidingWindow<T>::InitCUDAHandles() {
    // int gpu_count = 0;
    // cudaError_t error = cudaGetDeviceCount(&gpu_count);
    // printf("gpu_count : %d \n", gpu_count);

    cublasCreate_v2(&cublas_handle);
    cusolverDnCreate(&cusolver_dense_handle);
}

// ----------

template<typename T>
void SlidingWindow<T>::DestroyCUDAHandles() const {
    if(cublas_handle) {
        cublasDestroy(cublas_handle);
    }
    if(cusolver_dense_handle) {
        cusolverDnDestroy(cusolver_dense_handle);
    }
}

template<typename T>
void SlidingWindow<T>::SetStatesToInit() {
    cudaMemcpy(dev_ptr_states, dev_ptr_states_init, sizeof(T) * num_key_frames * 16, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_ex_para_0, dev_ptr_ex_para_0_init, sizeof(T) * 7, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_ex_para_1, dev_ptr_ex_para_1_init, sizeof(T) * 7, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_inv_depth, dev_ptr_inv_depth_init, shapes.num_elem_inv_depth * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_cur_td, dev_ptr_cur_td_init, shapes.num_elem_cur_td * sizeof(T), cudaMemcpyDeviceToDevice);

    P_init_vec.resize(num_key_frames);
    Q_init_vec.resize(num_key_frames);
    R_init_vec.resize(num_key_frames);
    V_init_vec.resize(num_key_frames);
    BiasAcc_init_vec.resize(num_key_frames);
    BiasGyr_init_vec.resize(num_key_frames);
    host_states_init.setZero();
    cudaMemcpy(host_states_init.data(), dev_ptr_states_init, sizeof(T) * num_key_frames * 16, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_key_frames; i++) {
        P_init_vec[i] = host_states_init.middleRows(i * 16 , 3);
        Q_init_vec[i] = Eigen::Quaternion<T>{ host_states_init(i * 16 + 3 + 3),    // real part goes first
                                              host_states_init(i * 16 + 3 + 0),
                                              host_states_init(i * 16 + 3 + 1),
                                              host_states_init(i * 16 + 3 + 2) }.normalized();
        R_init_vec[i] = Q_init_vec[i].toRotationMatrix();
        V_init_vec[i] = host_states_init.middleRows(i * 16 + 7 , 3);
        BiasAcc_init_vec[i] = host_states_init.middleRows(i * 16 + 10 , 3);
        BiasGyr_init_vec[i] = host_states_init.middleRows(i * 16 + 13 , 3);
    }

    host_Hprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp);
    host_Hprior.setZero();
    host_Bprior.resize(shapes.num_elem_Bpp, 1);
    host_Bprior.setZero();
    host_Jprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp);
    host_Jprior.setZero();
    host_Eprior.resize(shapes.num_elem_Bpp, 1);
    host_Eprior.setZero();

    LaunchProjFactorUpdateKernels();
    LaunchIMUFactorUpdateKernels();
}

template<typename T>
void SlidingWindow<T>::LaunchIMUFactorUpdateKernels() {
    IMUUpdateKernel(
        shapes.num_imu_factors,
        dev_ptr_states,
        imu_gpu_mem_pool.dev_ptr_set
    );
}

template<typename T>
void SlidingWindow<T>::LaunchProjFactorUpdateKernels() {
    Proj2F1CUpdateKernel(
        shapes.num_proj_2f1c_factors,
        dev_ptr_ex_para_0,
        dev_ptr_states,
        dev_ptr_inv_depth,
        dev_ptr_cur_td,
        proj_2f1c_gpu_mem_pool.dev_ptr_set
    );
    Proj2F2CUpdateKernel(
        shapes.num_proj_2f2c_factors,
        dev_ptr_ex_para_0,
        dev_ptr_ex_para_1,
        dev_ptr_states,
        dev_ptr_inv_depth,
        dev_ptr_cur_td,
        proj_2f2c_gpu_mem_pool.dev_ptr_set
    );
    Proj1F2CUpdateKernel(
        shapes.num_proj_1f2c_factors,
        dev_ptr_ex_para_0,
        dev_ptr_ex_para_1,
        dev_ptr_inv_depth,
        dev_ptr_cur_td,
        proj_1f2c_gpu_mem_pool.dev_ptr_set
    );
}

template<typename T>
void SlidingWindow<T>::LaunchAllFactorUpdateKernels() {
    AllUpdateKernels(
        dev_ptr_ex_para_0,
        dev_ptr_ex_para_1,
        dev_ptr_states,
        dev_ptr_inv_depth,
        dev_ptr_cur_td,
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        update_stream_set
    );
}

template<typename T>
void SlidingWindow<T>::LaunchAllBlockRangeKernels() {
    AllBlockRangeKernels(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        block_range_stream_set,
        use_imu,
        is_stereo
    );
}

template<typename T>
void SlidingWindow<T>::LaunchProjBlockRangeKernelsForMarg() {
    AllProjBlockRangeKernelsForMarg(
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        block_range_stream_set,
        is_stereo
    );
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::DevJacobianSetZero() const {
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_imu_factors * 90 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_imu_factors * 135 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_imu_factors * 90 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_imu_factors * 135 * sizeof(T));

    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_2f1c_factors * 2 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_4, 0, shapes.num_proj_2f1c_factors * 2 * sizeof(T));

    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_4, 0, shapes.num_proj_2f2c_factors * 2 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_5, 0, shapes.num_proj_2f2c_factors * 2 * sizeof(T));

    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_1f2c_factors * 2 * sizeof(T));
}

template<typename T>
void SlidingWindow<T>::DevJacobianSetZeroForMarg() const {
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_imu_factors * 90 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_imu_factors * 135 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_imu_factors * 90 * sizeof(T));
    cudaMemset(imu_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_imu_factors * 135 * sizeof(T));

    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_2f1c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_2f1c_factors * 2 * sizeof(T));
    cudaMemset(proj_2f1c_gpu_mem_pool.dev_ptr_set.jacobian_4, 0, shapes.num_proj_2f1c_factors * 2 * sizeof(T));

    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_2f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_4, 0, shapes.num_proj_2f2c_factors * 2 * sizeof(T));
    cudaMemset(proj_2f2c_gpu_mem_pool.dev_ptr_set.jacobian_5, 0, shapes.num_proj_2f2c_factors * 2 * sizeof(T));

    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_0, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_1, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_2, 0, shapes.num_proj_1f2c_factors * 12 * sizeof(T));
    cudaMemset(proj_1f2c_gpu_mem_pool.dev_ptr_set.jacobian_3, 0, shapes.num_proj_1f2c_factors * 2 * sizeof(T));
}

template<typename T>
void SlidingWindow<T>::DevHessianAndRHSSetZero() const {
    // cudaMemset(dev_ptr_Hpp, 0, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp);
    cudaMemcpy(dev_ptr_Hpp, dev_ptr_Hprior, sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyDeviceToDevice);
    cudaMemset(dev_ptr_Hpm, 0, sizeof(T) * shapes.num_rows_Hpm * shapes.num_cols_Hpm);
    cudaMemset(dev_ptr_Hmp, 0, sizeof(T) * shapes.num_rows_Hmp * shapes.num_cols_Hmp);
    cudaMemset(dev_ptr_Hmm_diag, 0, sizeof(T) * shapes.num_elem_Hmm_diag);

    // cudaMemset(dev_ptr_Bpp, 0, sizeof(T) * shapes.num_elem_Bpp);
    cudaMemcpy(dev_ptr_Bpp, dev_ptr_Bprior, sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyDeviceToDevice);
    cudaMemset(dev_ptr_Bmm, 0, sizeof(T) * shapes.num_elem_Bmm);
}

template<typename T>
void SlidingWindow<T>::DevHessianAndRHSSetZeroForMarg() const {
    cudaMemset(dev_ptr_Hpp_for_marg, 0, sizeof(T) * shapes.num_rows_Hpp_for_marg * shapes.num_cols_Hpp_for_marg);
    cudaMemset(dev_ptr_Hpm_for_marg, 0, sizeof(T) * shapes.num_rows_Hpm_for_marg * shapes.num_cols_Hpm_for_marg);
    cudaMemset(dev_ptr_Hmp_for_marg, 0, sizeof(T) * shapes.num_rows_Hmp_for_marg * shapes.num_cols_Hmp_for_marg);
    cudaMemset(dev_ptr_Hmm_diag_for_marg, 0, sizeof(T) * shapes.num_elem_Hmm_diag_for_marg);

    cudaMemset(dev_ptr_Bpp_for_marg, 0, sizeof(T) * shapes.num_elem_Bpp_for_marg);
    cudaMemset(dev_ptr_Bmm_for_marg, 0, sizeof(T) * shapes.num_elem_Bmm_for_marg);
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::LaunchJacResRobInfoKernels() {
    cudaError_t cuda_status;

    LaunchProjTempKernels();
    LaunchJacobianAndResidualKernels();
    LaunchRobustInfoKernels();
}

template<typename T>
void SlidingWindow<T>::LaunchProjTempKernels() {
    AllProjTempKernels(
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        proj_temp_stream_set,
        is_stereo
    );
}

template<typename T>
void SlidingWindow<T>::LaunchJacobianAndResidualKernels() {
    AllJacobianAndResidualKernels(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        jacobian_residual_stream_set,
        use_imu,
        is_stereo,
        pose_ex_0_is_fixed,
        pose_ex_1_is_fixed,
        cur_td_is_fixed
    );
}

template<typename T>
void SlidingWindow<T>::LaunchRobustInfoKernels() {
    AllRobustInfoKernels(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        robust_info_stream_set,
        use_imu,
        is_stereo
    );
}

// ------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::LaunchHessianAndRHSKernels() {
    AllHessianAndRHSKernels(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        hessian_rhs_stream_set,
        dev_ptr_Hpp,
        shapes.num_rows_Hpp,
        dev_ptr_Hpm,
        shapes.num_rows_Hpm,
        dev_ptr_Hmp,
        shapes.num_rows_Hmp,
        dev_ptr_Hmm_diag,
        dev_ptr_Bpp,
        dev_ptr_Bmm,
        use_imu,
        is_stereo,
        pose_ex_0_is_fixed,
        pose_ex_1_is_fixed,
        cur_td_is_fixed
    );
}

template<typename T>
void SlidingWindow<T>::LaunchHessianAndRHSKernelsForMarg() {
    AllProjTempKernelsForMarg(
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        proj_temp_stream_set,
        is_stereo
    );

    DevJacobianSetZero();

    AllJacobianAndResidualKernelsForMarg(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        jacobian_residual_stream_set,
        use_imu,
        is_stereo
    );

    AllRobustInfoKernelsForMarg(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        robust_info_stream_set,
        use_imu,
        is_stereo
    );

    AllHessianAndRHSKernelsForMarg(
        shapes.num_imu_factors,
        imu_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f1c_factors,
        proj_2f1c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_2f2c_factors,
        proj_2f2c_gpu_mem_pool.dev_ptr_set,
        shapes.num_proj_1f2c_factors,
        proj_1f2c_gpu_mem_pool.dev_ptr_set,
        hessian_rhs_stream_set,
        dev_ptr_Hpp_for_marg,
        shapes.num_rows_Hpp_for_marg,
        dev_ptr_Hpm_for_marg,
        shapes.num_rows_Hpm_for_marg,
        dev_ptr_Hmp_for_marg,
        shapes.num_rows_Hmp_for_marg,
        dev_ptr_Hmm_diag_for_marg,
        dev_ptr_Bpp_for_marg,
        dev_ptr_Bmm_for_marg,
        use_imu,
        is_stereo
    );
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::ComputeLambdaInitLMV2() {
    lambda.ComputeLambdaInitLMV2();
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::LaunchSolveDeltaKernels() {
    SolveDeltaKernels(
        cublas_handle,
        cusolver_dense_handle,
        shapes.num_rows_Hpp,
        shapes.num_cols_Hpp,
        dev_ptr_Hpp,
        shapes.num_elem_Hmm_diag,
        dev_ptr_Hmm_diag,
        shapes.num_rows_Hpm,
        shapes.num_cols_Hpm,
        dev_ptr_Hpm,
        shapes.num_rows_Hmp,
        shapes.num_cols_Hmp,
        dev_ptr_Hmp,
        shapes.num_elem_Bpp,
        dev_ptr_Bpp,
        shapes.num_elem_Bmm,
        dev_ptr_Bmm,
        dev_ptr_lambda,
        dev_ptr_Hpp_with_lambda,
        dev_ptr_Hmm_diag_with_lambda,
        dev_ptr_HpmHmmInv,
        dev_ptr_Hpp_schur,
        dev_ptr_Bpp_schur,
        dev_ptr_delta_Xpp,
        dev_ptr_delta_inv_depth
    );

    // ------------------------------------------------------------------------------------------------------------------------

    int offset = 0;
    int nelem  = 0;

    offset = 0;
    nelem  = shapes.num_elem_delta_ex_para;
    if(pose_ex_0_is_fixed) {
        cudaMemset(dev_ptr_delta_Xpp + offset, 0, nelem * sizeof(T));
    }
    cudaMemcpy(dev_ptr_delta_ex_para_0, dev_ptr_delta_Xpp + offset, nelem * sizeof(T), cudaMemcpyDeviceToDevice);

    offset = shapes.num_elem_delta_ex_para;
    nelem  = shapes.num_elem_delta_ex_para;
    if(pose_ex_1_is_fixed) {
        cudaMemset(dev_ptr_delta_Xpp + offset, 0, nelem * sizeof(T));
    }
    cudaMemcpy(dev_ptr_delta_ex_para_1, dev_ptr_delta_Xpp + offset, nelem * sizeof(T), cudaMemcpyDeviceToDevice);

    offset = shapes.num_elem_delta_ex_para + shapes.num_elem_delta_ex_para;
    nelem  = shapes.num_elem_delta_cur_td;
    if(cur_td_is_fixed) {
        cudaMemset(dev_ptr_delta_Xpp + offset, 0, nelem * sizeof(T));
    }
    cudaMemcpy(dev_ptr_delta_cur_td, dev_ptr_delta_Xpp + offset, nelem * sizeof(T), cudaMemcpyDeviceToDevice);

    offset = shapes.num_elem_delta_ex_para + shapes.num_elem_delta_ex_para + 1;
    nelem  = shapes.num_elem_delta_states;
    if(pose_0_is_fixed) {
        cudaMemset(dev_ptr_delta_Xpp + offset, 0, 6 * sizeof(T));
    }
    cudaMemcpy(dev_ptr_delta_states, dev_ptr_delta_Xpp + offset, nelem * sizeof(T), cudaMemcpyDeviceToDevice);
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::BackupStatesAndInvDepth() {
    cudaMemset(dev_ptr_inv_depth_backup, 0, shapes.max_num_elem_inv_depth * sizeof(T));

    cudaMemcpy(dev_ptr_ex_para_0_backup, dev_ptr_ex_para_0, shapes.num_elem_ex_para * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_ex_para_1_backup, dev_ptr_ex_para_1, shapes.num_elem_ex_para * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_states_backup, dev_ptr_states, shapes.num_elem_states * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_inv_depth_backup, dev_ptr_inv_depth, shapes.num_elem_inv_depth * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_cur_td_backup, dev_ptr_cur_td, shapes.num_elem_cur_td * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
void SlidingWindow<T>::RollBackStatesAndInvDepth() {
    cudaMemcpy(dev_ptr_ex_para_0, dev_ptr_ex_para_0_backup, shapes.num_elem_ex_para * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_ex_para_1, dev_ptr_ex_para_1_backup, shapes.num_elem_ex_para * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_states, dev_ptr_states_backup, shapes.num_elem_states * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_inv_depth, dev_ptr_inv_depth_backup, shapes.num_elem_inv_depth * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ptr_cur_td, dev_ptr_cur_td_backup, shapes.num_elem_cur_td * sizeof(T), cudaMemcpyDeviceToDevice);
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::LaunchUpdateKernels() {
    LaunchStatesAddDeltaKernels();
    LaunchInvDepthAddDeltaKernels();
    LaunchAllFactorUpdateKernels();
}

template<typename T>
void SlidingWindow<T>::LaunchStatesAddDeltaKernels() const {
    StatesAddDeltaKernel(
        shapes.num_key_frames,
        dev_ptr_delta_ex_para_0,
        dev_ptr_delta_ex_para_1,
        dev_ptr_delta_states,
        dev_ptr_delta_cur_td,
        dev_ptr_ex_para_0,
        dev_ptr_ex_para_1,
        dev_ptr_states,
        dev_ptr_cur_td
    );
}

template<typename T>
void SlidingWindow<T>::LaunchInvDepthAddDeltaKernels() const {
    InvDepthAddDeltaKernel(
        shapes.num_world_points,
        dev_ptr_delta_inv_depth,
        dev_ptr_inv_depth
    );
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::BackupBpriorAndEprior() {
    host_Bprior_backup.resizeLike(host_Bprior);
    host_Eprior_backup.resizeLike(host_Eprior);
    cudaMemcpy(host_Bprior_backup.data(), dev_ptr_Bprior, sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Eprior_backup.data(), dev_ptr_Eprior, sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyDeviceToHost);
}

template<typename T>
void SlidingWindow<T>::UpdateBpriorAndEprior() const {
    cublasStatus_t cublas_status;
    static T alpha;
    static T beta;

    alpha = -1.0;
    beta = 1.0;
    cublas_status = cublas_gemv_v2<T>(
            cublas_handle,
            CUBLAS_OP_N,
            shapes.num_rows_Hpp,
            shapes.num_cols_Hpp,
            &alpha,
            dev_ptr_Hprior,
            shapes.num_rows_Hpp,
            dev_ptr_delta_Xpp,
            1,
            &beta,
            dev_ptr_Bprior,
            1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    if(decompose_prior) {
        alpha = 1.0;
        beta = 1.0;
        cublas_status = cublas_gemv_v2<T>(
            cublas_handle,
            CUBLAS_OP_N,
            shapes.num_rows_Hpp,
            shapes.num_cols_Hpp,
            &alpha,
            dev_ptr_Jprior,
            shapes.num_rows_Hpp,
            dev_ptr_delta_Xpp,
            1,
            &beta,
            dev_ptr_Eprior,
            1
        );
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    }
}

template<typename T>
void SlidingWindow<T>::RollBackBpriorAndEprior() {
    cudaMemcpy(dev_ptr_Bprior, host_Bprior_backup.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Eprior, host_Eprior_backup.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
bool SlidingWindow<T>::IsGoodStepInLM() {
    return lambda.IsGoodStepInLM();
}

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void SlidingWindow<T>::LaunchMarginalizeInvDepthKernels() {
    if(shapes.num_world_points_for_marg == 0) {
        return;
    }

    MargInvDepthKernels(
        cublas_handle,
        shapes.num_rows_Hpp_for_marg,
        shapes.num_cols_Hpp_for_marg,
        dev_ptr_Hpp_for_marg,
        shapes.num_elem_Hmm_diag_for_marg,
        dev_ptr_Hmm_diag_for_marg,
        shapes.num_rows_Hpm_for_marg,
        shapes.num_cols_Hpm_for_marg,
        dev_ptr_Hpm_for_marg,
        shapes.num_rows_Hmp_for_marg,
        shapes.num_cols_Hmp_for_marg,
        dev_ptr_Hmp_for_marg,
        shapes.num_elem_Bpp_for_marg,
        dev_ptr_Bpp_for_marg,
        shapes.num_elem_Bmm_for_marg,
        dev_ptr_Bmm_for_marg
    );
}

template<typename T>
void SlidingWindow<T>::MargKeyFrameWithPriorDecomp() {
    host_Bprior.resize(shapes.num_elem_Bpp, 1);
    host_Bprior.setZero();

    cudaMemcpy(host_Bprior.data(), dev_ptr_Bprior, sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyDeviceToHost);

    MargKeyFrameKernelsWithPriorDecomp(
        marg_keyframe_idx,
        cublas_handle,
        cusolver_dense_handle,
        shapes.num_rows_Hpp_for_marg,
        shapes.num_cols_Hpp_for_marg,
        dev_ptr_Hpp_for_marg,
        shapes.num_elem_Bpp_for_marg,
        dev_ptr_Bpp_for_marg,
        shapes.num_rows_Hpp_for_marg,
        shapes.num_cols_Hpp_for_marg,
        dev_ptr_Hprior,
        host_Bprior,
        shapes.num_rows_H11,
        shapes.num_cols_H11,
        dev_ptr_H11,
        shapes.num_rows_H12,
        shapes.num_cols_H12,
        dev_ptr_H12,
        shapes.num_rows_H21,
        shapes.num_cols_H21,
        dev_ptr_H21,
        shapes.num_rows_H22,
        shapes.num_cols_H22,
        dev_ptr_H22,
        shapes.num_elem_B11,
        dev_ptr_B11,
        shapes.num_elem_B22,
        dev_ptr_B22,
        dev_ptr_Hprior_eigenvec,
        dev_ptr_Hprior_eigenval
    );

    host_Hprior_eigenvec.resize(shapes.num_rows_H11, shapes.num_cols_H11);
    host_Hprior_eigenval.resize(shapes.num_rows_H11, 1);
    cudaMemcpy(host_Hprior_eigenvec.data(), dev_ptr_Hprior_eigenvec, sizeof(T) * shapes.num_rows_H11 * shapes.num_cols_H11, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Hprior_eigenval.data(), dev_ptr_Hprior_eigenval, sizeof(T) * shapes.num_rows_H11, cudaMemcpyDeviceToHost);

    host_Hprior_eigenval = (host_Hprior_eigenval.array() > 1e-8).select(host_Hprior_eigenval.array(), 0);
    host_Hprior_eigenval_sqrt = ((host_Hprior_eigenval.array() > 1e-8).select(host_Hprior_eigenval.array(), 0)).cwiseSqrt();
    host_Hprior_eigenval_sqrt_inv = ((host_Hprior_eigenval.array() > 1e-8).select(host_Hprior_eigenval.array().inverse(), 0)).cwiseSqrt();

    host_Hprior_temp = host_Hprior_eigenval_sqrt.asDiagonal() * host_Hprior_eigenvec.transpose();   // J
    host_Hprior_temp = host_Hprior_temp.transpose() * host_Hprior_temp;                             // JTJ
    host_Hprior_temp = (host_Hprior_temp.array().abs() > 1e-9).select(host_Hprior_temp.array(), 0.0);
    host_Hprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp); host_Hprior.setZero();
    host_Hprior.block(0, 0, shapes.num_rows_H11, shapes.num_cols_H11) = host_Hprior_temp;
    // if(pose_ex_0_is_fixed) {
    //     host_Hprior.middleRows(0, 6).setZero();
    //     host_Hprior.middleCols(0, 6).setZero();
    // }
    // if(pose_ex_1_is_fixed) {
    //     host_Hprior.middleRows(6, 6).setZero();
    //     host_Hprior.middleCols(6, 6).setZero();
    // }
    // if(cur_td_is_fixed) {
    //     host_Hprior.middleRows(12, 1).setZero();
    //     host_Hprior.middleCols(12, 1).setZero();
    // }
    if(pose_0_is_fixed) {
        host_Hprior.middleRows(13, 6).setZero();
        host_Hprior.middleCols(13, 6).setZero();
    }

    host_Bprior_temp.resize(shapes.num_rows_H11, 1);
    cudaMemcpy(host_Bprior_temp.data(), dev_ptr_B11, sizeof(T) * shapes.num_rows_H11, cudaMemcpyDeviceToHost);
    host_Bprior.resize(shapes.num_elem_Bpp, 1); host_Bprior.setZero();
    host_Bprior.block(0, 0, shapes.num_elem_B11, 1) = host_Bprior_temp;
    // if(pose_ex_0_is_fixed) {
    //     host_Bprior.middleRows(0, 6).setZero();
    // }
    // if(pose_ex_1_is_fixed) {
    //     host_Bprior.middleRows(6, 6).setZero();
    // }
    // if(cur_td_is_fixed) {
    //     host_Bprior.middleRows(12, 1).setZero();
    // }
    if(pose_0_is_fixed) {
        host_Bprior.middleRows(13, 6).setZero();
    }

    host_Jprior_temp = host_Hprior_eigenval_sqrt.asDiagonal() * host_Hprior_eigenvec.transpose();
    host_Eprior_temp = -1.0 * host_Hprior_eigenval_sqrt_inv.asDiagonal() * host_Hprior_eigenvec.transpose() * host_Bprior_temp;
    assert(host_Jprior_temp.rows() == shapes.num_rows_H11 && host_Jprior_temp.cols() == shapes.num_cols_H11);
    assert(host_Eprior_temp.rows() == shapes.num_rows_H11 && host_Eprior_temp.cols() == 1);

    host_Jprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp); host_Jprior.setZero();
    host_Jprior.block(0, 0, shapes.num_rows_H11, shapes.num_cols_H11) = host_Jprior_temp;

    host_Eprior.resize(shapes.num_elem_Bpp, 1); host_Eprior.setZero();
    host_Eprior.block(0, 0, shapes.num_elem_B11, 1) = host_Eprior_temp;

    cudaMemcpy(dev_ptr_Hprior, host_Hprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Bprior, host_Bprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Jprior, host_Jprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Bprior, host_Eprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
}

template<typename T>
void SlidingWindow<T>::MargKeyFrameWithoutPriorDecomp() {
    host_Bprior.resize(shapes.num_elem_Bpp, 1);
    host_Bprior.setZero();

    cudaMemcpy(host_Bprior.data(), dev_ptr_Bprior, sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyDeviceToHost);

    MargKeyFrameKernelsWithoutPriorDecomp(
            marg_keyframe_idx,
            cublas_handle,
            cusolver_dense_handle,
            shapes.num_rows_Hpp_for_marg,
            shapes.num_cols_Hpp_for_marg,
            dev_ptr_Hpp_for_marg,
            shapes.num_elem_Bpp_for_marg,
            dev_ptr_Bpp_for_marg,
            shapes.num_rows_Hpp_for_marg,
            shapes.num_cols_Hpp_for_marg,
            dev_ptr_Hprior,
            host_Bprior,
            shapes.num_rows_H11,
            shapes.num_cols_H11,
            dev_ptr_H11,
            shapes.num_rows_H12,
            shapes.num_cols_H12,
            dev_ptr_H12,
            shapes.num_rows_H21,
            shapes.num_cols_H21,
            dev_ptr_H21,
            shapes.num_rows_H22,
            shapes.num_cols_H22,
            dev_ptr_H22,
            shapes.num_elem_B11,
            dev_ptr_B11,
            shapes.num_elem_B22,
            dev_ptr_B22
    );

    host_Hprior_temp.resize(shapes.num_rows_H11, shapes.num_cols_H11); host_Hprior_temp.setZero();
    cudaMemcpy(host_Hprior_temp.data(), dev_ptr_H11, sizeof(T) * shapes.num_rows_H11 * shapes.num_cols_H11, cudaMemcpyDeviceToHost);
    host_Hprior_temp = (host_Hprior_temp.array().abs() > 1e-9).select(host_Hprior_temp.array(), 0.0);
    host_Hprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp); host_Hprior.setZero();
    host_Hprior.block(0, 0, shapes.num_rows_H11, shapes.num_cols_H11) = host_Hprior_temp;
    // if(pose_ex_0_is_fixed) {
    //     host_Hprior.middleRows(0, 6).setZero();
    //     host_Hprior.middleCols(0, 6).setZero();
    // }
    // if(pose_ex_1_is_fixed) {
    //     host_Hprior.middleRows(6, 6).setZero();
    //     host_Hprior.middleCols(6, 6).setZero();
    // }
    // if(cur_td_is_fixed) {
    //     host_Hprior.middleRows(12, 1).setZero();
    //     host_Hprior.middleCols(12, 1).setZero();
    // }
    if(pose_0_is_fixed) {
        host_Hprior.middleRows(13, 6).setZero();
        host_Hprior.middleCols(13, 6).setZero();
    }

    host_Bprior_temp.resize(shapes.num_rows_H11, 1);
    cudaMemcpy(host_Bprior_temp.data(), dev_ptr_B11, sizeof(T) * shapes.num_rows_H11, cudaMemcpyDeviceToHost);
    host_Bprior.resize(shapes.num_elem_Bpp, 1); host_Bprior.setZero();
    host_Bprior.block(0, 0, shapes.num_elem_B11, 1) = host_Bprior_temp;
    // if(pose_ex_0_is_fixed) {
    //     host_Bprior.middleRows(0, 6).setZero();
    // }
    // if(pose_ex_1_is_fixed) {
    //     host_Bprior.middleRows(6, 6).setZero();
    // }
    // if(cur_td_is_fixed) {
    //     host_Bprior.middleRows(12, 1).setZero();
    // }
    if(pose_0_is_fixed) {
        host_Bprior.middleRows(13, 6).setZero();
    }

    host_Jprior.resize(shapes.num_rows_Hpp, shapes.num_cols_Hpp); host_Jprior.setZero();
    host_Eprior.resize(shapes.num_elem_Bpp, 1); host_Eprior.setZero();

    cudaMemcpy(dev_ptr_Hprior, host_Hprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Bprior, host_Bprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Jprior, host_Jprior.data(), sizeof(T) * shapes.num_rows_Hpp * shapes.num_cols_Hpp, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_Bprior, host_Eprior.data(), sizeof(T) * shapes.num_elem_Bpp, cudaMemcpyHostToDevice);
}

template<typename T>
void SlidingWindow<T>::LaunchMarginalizeKeyFrameKernels() {
    if(decompose_prior) {
        MargKeyFrameWithPriorDecomp();
    } else {
        MargKeyFrameWithoutPriorDecomp();
    }

    have_prior = true;
}

template<typename T>
void SlidingWindow<T>::AlignYawToFirstFrame() {
    P_vec.resize(num_key_frames);
    Q_vec.resize(num_key_frames);
    R_vec.resize(num_key_frames);
    V_vec.resize(num_key_frames);
    BiasAcc_vec.resize(num_key_frames);
    BiasGyr_vec.resize(num_key_frames);

    host_inv_depth.resize(shapes.num_elem_inv_depth, 1); host_inv_depth.setZero();
    cudaMemcpy(host_inv_depth.data(), dev_ptr_inv_depth, sizeof(T) * shapes.num_elem_inv_depth, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ex_para_0.data(), dev_ptr_ex_para_0, sizeof(T) * shapes.num_elem_ex_para, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ex_para_1.data(), dev_ptr_ex_para_1, sizeof(T) * shapes.num_elem_ex_para, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cur_td.data(), dev_ptr_cur_td, sizeof(T) * shapes.num_elem_cur_td, cudaMemcpyDeviceToHost);

    host_states.resize(num_key_frames * 16, 1); host_states.setZero();
    cudaMemcpy(host_states.data(), dev_ptr_states, sizeof(T) * num_key_frames * 16, cudaMemcpyDeviceToHost);

    for(int k = 0; k < num_key_frames; k++) {
        P_vec[k] = host_states.middleRows(k * 16 , 3);
        Q_vec[k] = Eigen::Quaternion<T>{ host_states(k * 16 + 3 + 3),    // real part goes first
                                         host_states(k * 16 + 3 + 0),
                                         host_states(k * 16 + 3 + 1),
                                         host_states(k * 16 + 3 + 2) }.normalized();
        R_vec[k] = Q_vec[k].toRotationMatrix();
        V_vec[k] = host_states.middleRows(k * 16 + 7 , 3);
        BiasAcc_vec[k] = host_states.middleRows(k * 16 + 10 , 3);
        BiasGyr_vec[k] = host_states.middleRows(k * 16 + 13 , 3);
    }

    Eigen::Matrix<T, 3, 1> origin_R0 = R2ypr(R_init_vec[0]);
    Eigen::Matrix<T, 3, 1> origin_P0 = P_init_vec[0];
    Eigen::Matrix<T, 3, 1> origin_R00 = R2ypr(R_vec[0]);
    T y_diff = origin_R0.x() - origin_R00.x();
    Eigen::Matrix<T, 3, 3> rot_diff = ypr2R(Eigen::Matrix<T, 3, 1>(y_diff, 0, 0));

    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        printf("euler singular point!");
        rot_diff = R_init_vec[0] * R_vec[0].transpose();
    }

    for(int k = 0; k < num_key_frames; k++) {
        P_vec[k] = rot_diff * (P_vec[k] - P_vec[0]) + origin_P0;
        R_vec[k] = rot_diff * R_vec[k];
        Q_vec[k] = Eigen::Quaternion<T>(R_vec[k]).normalized();
        V_vec[k] = rot_diff * V_vec[k];
    }

    for(int k = 0; k < num_key_frames; k++) {
        host_states.middleRows(k * 16, 3) = P_vec[k];
        host_states(k * 16 + 3 + 0) = Q_vec[k].x();
        host_states(k * 16 + 3 + 1) = Q_vec[k].y();
        host_states(k * 16 + 3 + 2) = Q_vec[k].z();
        host_states(k * 16 + 3 + 3) = Q_vec[k].w();
        host_states.middleRows(k * 16 +  7, 3) = V_vec[k];
        host_states.middleRows(k * 16 + 10, 3) = BiasAcc_vec[k];
        host_states.middleRows(k * 16 + 13, 3) = BiasGyr_vec[k];
    }

    cudaMemcpy(dev_ptr_states, host_states.data(), sizeof(T) * num_key_frames * 16, cudaMemcpyHostToDevice);
}

template<typename T>
void SlidingWindow<T>::SetPriorZero() {
    host_Jprior.resize(num_key_frames * 15 + 13, num_key_frames * 15 + 13);
    host_Jprior.setZero();
    host_Eprior.resize(num_key_frames * 15 + 13, 1);
    host_Eprior.setZero();
    host_Hprior.resize(num_key_frames * 15 + 13, num_key_frames * 15 + 13);
    host_Hprior.setZero();
    host_Bprior.resize(num_key_frames * 15 + 13, 1);
    host_Bprior.setZero();
}

template<typename T>
void SlidingWindow<T>::SetInitStates(const Eigen::MatrixXd& init_states) {
    assert((init_states.rows() == num_key_frames * 16) && (init_states.cols() == 1));

    host_states = init_states.cast<T>();
    host_states_backup = host_states;
    host_states_init = host_states;
}

template<typename T>
void SlidingWindow<T>::PrintStates() {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_latest_states = host_states.middleRows((num_key_frames - 1) * 16, 7).transpose();
    printf(
        "latest_states : %f , %f , %f , %f , %f , %f , %f \n", 
        host_latest_states(0), host_latest_states(1), host_latest_states(2),
        host_latest_states(3), host_latest_states(4), host_latest_states(5), host_latest_states(6)
    );
}

template<typename T>
bool SlidingWindow<T>::NeedToMarginalize() {
    assert((marg_keyframe_idx == 0) || (marg_keyframe_idx == num_key_frames - 2));
    
    if(marg_keyframe_idx == 0) {
        return true;
    } else {
        return prev_keyframe_idx_in_marg.count(num_key_frames - 1) > 0;
    }
}

// ----------

// instantiation
template class SlidingWindow<double>;
// template class SlidingWindow<float>;    // do not use float!!!

} // namespace VINS_FUSION_CUDA_BA

