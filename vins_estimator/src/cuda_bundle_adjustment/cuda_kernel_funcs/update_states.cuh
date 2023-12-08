//
// Created by lmf on 23-7-28.
//

#ifndef CUDA_BA_UPDATE_STATES_CUH
#define CUDA_BA_UPDATE_STATES_CUH

#include <cstdio>
#include <string>
#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <cuda_runtime.h>

#include "device_utils.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void imu_update(
    int num_imu_factors,
    const T* input_states,
    IMUFactorDevPtrSet<T> dev_ptr_set
);

// ----------

template<typename T>
__global__ void proj_2f1c_update(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F1CFactorDevPtrSet<T> dev_ptr_set
);

template<typename T>
__global__ void proj_2f2c_update(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set
);

template<typename T>
__global__ void proj_1f2c_update(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set
);

// ----------

template<typename T>
void StatesAddDeltaKernel(
    int num_key_frames,
    const T* input_delta_ex_para_0,    // length = 6
    const T* input_delta_ex_para_1,    // length = 6
    const T* input_delta_states,     // stride = 15
    const T* input_delta_cur_td,
    T* output_ex_para_0,    // length = 7
    T* output_ex_para_1,    // length = 7
    T* output_states,      // stride = 16
    T* output_cur_td
);

template<typename T>
void InvDepthAddDeltaKernel(
    int num_world_points,
    const T* input_delta_inv_depth,
    T* output_inv_depth
);

template<typename T>
void IMUUpdateKernel(
    int num_imu_factors,
    const T* input_states,
    IMUFactorDevPtrSet<T>& dev_ptr_set
);

template<typename T>
void Proj2F1CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F1CFactorDevPtrSet<T>& dev_ptr_set
);

template<typename T>
void Proj2F2CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F2CFactorDevPtrSet<T>& dev_ptr_set
);

template<typename T>
void Proj1F2CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj1F2CFactorDevPtrSet<T>& dev_ptr_set
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_UPDATE_STATES_CUH
