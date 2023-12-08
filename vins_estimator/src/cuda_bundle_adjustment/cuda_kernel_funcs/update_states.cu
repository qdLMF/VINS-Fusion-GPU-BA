//
// Created by lmf on 23-7-28.
//

#include "update_states.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void states_add_delta(
    int num_key_frames,
    const T* input_delta_ex_para_0,     // length = 6
    const T* input_delta_ex_para_1,     // length = 6
    const T* input_delta_states,        // stride = 15
    const T* input_delta_cur_td,
    T* output_ex_para_0,                // length = 7
    T* output_ex_para_1,                // length = 7
    T* output_states,                   // stride = 16
    T* output_cur_td
) {
    unsigned int idx_frame = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_frame >= num_key_frames)
        return;

    Eigen::Matrix<T, 3, 1> delta_p;
    delta_p(0) = input_delta_states[idx_frame * 15 + 0];
    delta_p(1) = input_delta_states[idx_frame * 15 + 1];
    delta_p(2) = input_delta_states[idx_frame * 15 + 2];

    Eigen::Matrix<T, 3, 1> delta_theta;
    delta_theta(0) = input_delta_states[idx_frame * 15 + 3 + 0];
    delta_theta(1) = input_delta_states[idx_frame * 15 + 3 + 1];
    delta_theta(2) = input_delta_states[idx_frame * 15 + 3 + 2];
    Eigen::Quaternion<T> delta_q = UtilityDeltaQ(delta_theta);

    Eigen::Matrix<T, 3, 1> delta_v;
    delta_v(0) = input_delta_states[idx_frame * 15 + 6 + 0];
    delta_v(1) = input_delta_states[idx_frame * 15 + 6 + 1];
    delta_v(2) = input_delta_states[idx_frame * 15 + 6 + 2];

    Eigen::Matrix<T, 3, 1> delta_bias_acc;
    delta_bias_acc(0) = input_delta_states[idx_frame * 15 + 9 + 0];
    delta_bias_acc(1) = input_delta_states[idx_frame * 15 + 9 + 1];
    delta_bias_acc(2) = input_delta_states[idx_frame * 15 + 9 + 2];

    Eigen::Matrix<T, 3, 1> delta_bias_gyr;
    delta_bias_gyr(0) = input_delta_states[idx_frame * 15 + 12 + 0];
    delta_bias_gyr(1) = input_delta_states[idx_frame * 15 + 12 + 1];
    delta_bias_gyr(2) = input_delta_states[idx_frame * 15 + 12 + 2];

    Eigen::Matrix<T, 3, 1> p;
    for (int i = 0; i < p.size(); i++) {
        p(i) = *(output_states + 16 * idx_frame + 0 + i);
    }
    p = p + delta_p;
    for (int i = 0; i < p.size(); i++) {
        *(output_states + 16 * idx_frame + 0 + i) = p(i);
    }

    Eigen::Quaternion<T> q;
    for (int i = 0; i < q.coeffs().size(); i++) {
        q.coeffs()(i) = *(output_states + 16 * idx_frame + 3 + i);
    }
    q = (q * delta_q).normalized();
    for (int i = 0; i < q.coeffs().size(); i++) {
        *(output_states + 16 * idx_frame + 3 + i) = q.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> v;
    for (int i = 0; i < v.size(); i++) {
        v(i) = *(output_states + 16 * idx_frame + 7 + i);
    }
    v = v + delta_v;
    for (int i = 0; i < v.size(); i++) {
        *(output_states + 16 * idx_frame + 7 + i) = v(i);
    }

    Eigen::Matrix<T, 3, 1> bias_acc;
    for (int i = 0; i < bias_acc.size(); i++) {
        bias_acc(i) = *(output_states + 16 * idx_frame + 10 + i);
    }

    bias_acc = bias_acc + delta_bias_acc;
    for (int i = 0; i < bias_acc.size(); i++) {
        *(output_states + 16 * idx_frame + 10 + i) = bias_acc(i);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr;
    for (int i = 0; i < bias_gyr.size(); i++) {
        bias_gyr(i) = *(output_states + 16 * idx_frame + 13 + i);
    }
    bias_gyr = bias_gyr + delta_bias_gyr;
    for (int i = 0; i < bias_gyr.size(); i++) {
        *(output_states + 16 * idx_frame + 13 + i) = bias_gyr(i);
    }

    if(idx_frame == 0) {
        Eigen::Matrix<T, 3, 1> delta_p_ex_0;
        delta_p_ex_0(0) = input_delta_ex_para_0[0];
        delta_p_ex_0(1) = input_delta_ex_para_0[1];
        delta_p_ex_0(2) = input_delta_ex_para_0[2];

        Eigen::Matrix<T, 3, 1> delta_theta_ex_0;
        delta_theta_ex_0(0) = input_delta_ex_para_0[3 + 0];
        delta_theta_ex_0(1) = input_delta_ex_para_0[3 + 1];
        delta_theta_ex_0(2) = input_delta_ex_para_0[3 + 2];
        Eigen::Quaternion<T> delta_q_ex_0 = UtilityDeltaQ<T>(delta_theta_ex_0);

        Eigen::Matrix<T, 3, 1> p_ex_0;
        for (int i = 0; i < p_ex_0.size(); i++) {
            p_ex_0(i) = *(output_ex_para_0 + 0 + i);
        }
        p_ex_0 = p_ex_0 + delta_p_ex_0;
        for (int i = 0; i < p_ex_0.size(); i++) {
            *(output_ex_para_0 + 0 + i) = p_ex_0(i);
        }

        Eigen::Quaternion<T> q_ex_0;
        for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
            q_ex_0.coeffs()(i) = *(output_ex_para_0 + 3 + i);
        }
        q_ex_0 = (q_ex_0 * delta_q_ex_0).normalized();
        for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
            *(output_ex_para_0 + 3 + i) = q_ex_0.coeffs()(i);
        }

        // ----------

        Eigen::Matrix<T, 3, 1> delta_p_ex_1;
        delta_p_ex_1(0) = input_delta_ex_para_1[0];
        delta_p_ex_1(1) = input_delta_ex_para_1[1];
        delta_p_ex_1(2) = input_delta_ex_para_1[2];

        Eigen::Matrix<T, 3, 1> delta_theta_ex_1;
        delta_theta_ex_1(0) = input_delta_ex_para_1[3 + 0];
        delta_theta_ex_1(1) = input_delta_ex_para_1[3 + 1];
        delta_theta_ex_1(2) = input_delta_ex_para_1[3 + 2];
        Eigen::Quaternion<T> delta_q_ex_1 = UtilityDeltaQ(delta_theta_ex_1);

        Eigen::Matrix<T, 3, 1> p_ex_1;
        for (int i = 0; i < p_ex_1.size(); i++) {
            p_ex_1(i) = *(output_ex_para_1 + 0 + i);
        }
        p_ex_1 = p_ex_1 + delta_p_ex_1;
        for (int i = 0; i < p_ex_1.size(); i++) {
            *(output_ex_para_1 + 0 + i) = p_ex_1(i);
        }

        Eigen::Quaternion<T> q_ex_1;
        for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
            q_ex_1.coeffs()(i) = *(output_ex_para_1 + 3 + i);
        }
        q_ex_1 = (q_ex_1 * delta_q_ex_1).normalized();
        for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
            *(output_ex_para_1 + 3 + i) = q_ex_1.coeffs()(i);
        }

        // ----------

        (*output_cur_td) += (*input_delta_cur_td);
    }
}
// instantiation
template __global__ void states_add_delta<double>(
    int num_key_frames,
    const double* input_delta_ex_para_0,    // length = 6
    const double* input_delta_ex_para_1,    // length = 6
    const double* input_delta_states,       // stride = 15
    const double* input_delta_cur_td,
    double* output_ex_para_0,               // length = 7
    double* output_ex_para_1,               // length = 7
    double* output_states,                  // stride = 16
    double* output_cur_td
);
template __global__ void states_add_delta<float>(
    int num_key_frames,
    const float* input_delta_ex_para_0,     // length = 6
    const float* input_delta_ex_para_1,     // length = 6
    const float* input_delta_states,        // stride = 15
    const float* input_delta_cur_td,
    float* output_ex_para_0,                // length = 7
    float* output_ex_para_1,                // length = 7
    float* output_states,                   // stride = 16
    float* output_cur_td
);

// ----------

template<typename T>
__global__ void inv_depth_add_delta(
    int num_world_points,
    const T* input_delta_inv_depth,
    T* output_inv_depth
) {
    unsigned int idx_inv_depth = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_inv_depth >= num_world_points)
        return;

    *(output_inv_depth + idx_inv_depth) += *(input_delta_inv_depth + idx_inv_depth);
}
// instantiation
template __global__ void inv_depth_add_delta<double>(
    int num_world_points,
    const double* input_delta_inv_depth,
    double* output_inv_depth
);
template __global__ void inv_depth_add_delta<float>(
    int num_world_points,
    const float* input_delta_inv_depth,
    float* output_inv_depth
);

// ----------

template<typename T>
__global__ void imu_update(
    int num_imu_factors,
    const T* input_states,
    IMUFactorDevPtrSet<T> dev_ptr_set
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_factor >= num_imu_factors)
        return;

    // ********** start : read inputs **********

    int imu_idx_i = *(dev_ptr_set.idx_i + idx_factor);
    int imu_idx_j = *(dev_ptr_set.idx_j + idx_factor);

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    Eigen::Matrix<T, 3, 1> p_i;
    for (int i = 0; i < p_i.size(); i++) {
        p_i(i) = *(input_states + 16 * imu_idx_i + 0 + i);
    }
    for (int i = 0; i < p_i.size(); i++) {
        *(dev_ptr_set.p_i + i * num_imu_factors + idx_factor) = p_i(i);
    }

    Eigen::Quaternion<T> q_i;
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(input_states + 16 * imu_idx_i + 3 + i);
    }
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        *(dev_ptr_set.q_i + i * num_imu_factors + idx_factor) = q_i.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> v_i;
    for (int i = 0; i < v_i.size(); i++) {
        v_i(i) = *(input_states + 16 * imu_idx_i + 7 + i);
    }
    for (int i = 0; i < v_i.size(); i++) {
        *(dev_ptr_set.v_i + i * num_imu_factors + idx_factor) = v_i(i);
    }

    Eigen::Matrix<T, 3, 1> bias_acc_i;
    for (int i = 0; i < bias_acc_i.size(); i++) {
        bias_acc_i(i) = *(input_states + 16 * imu_idx_i + 10 + i);
    }
    for (int i = 0; i < bias_acc_i.size(); i++) {
        *(dev_ptr_set.bias_acc_i + i * num_imu_factors + idx_factor) = bias_acc_i(i);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_i;
    for (int i = 0; i < bias_gyr_i.size(); i++) {
        bias_gyr_i(i) = *(input_states + 16 * imu_idx_i + 13 + i);
    }
    for (int i = 0; i < bias_gyr_i.size(); i++) {
        *(dev_ptr_set.bias_gyr_i + i * num_imu_factors + idx_factor) = bias_gyr_i(i);
    }

    Eigen::Matrix<T, 3, 1> p_j;
    for (int i = 0; i < p_j.size(); i++) {
        p_j(i) = *(input_states + 16 * imu_idx_j + 0 + i);
    }
    for (int i = 0; i < p_j.size(); i++) {
        *(dev_ptr_set.p_j + i * num_imu_factors + idx_factor) = p_j(i);
    }

    Eigen::Quaternion<T> q_j;
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(input_states + 16 * imu_idx_j + 3 + i);
    }
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        *(dev_ptr_set.q_j + i * num_imu_factors + idx_factor) = q_j.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> v_j;
    for (int i = 0; i < v_j.size(); i++) {
        v_j(i) = *(input_states + 16 * imu_idx_j + 7 + i);
    }
    for (int i = 0; i < v_j.size(); i++) {
        *(dev_ptr_set.v_j + i * num_imu_factors + idx_factor) = v_j(i);
    }

    Eigen::Matrix<T, 3, 1> bias_acc_j;
    for (int i = 0; i < bias_acc_j.size(); i++) {
        bias_acc_j(i) = *(input_states + 16 * imu_idx_j + 10 + i);
    }
    for (int i = 0; i < bias_acc_j.size(); i++) {
        *(dev_ptr_set.bias_acc_j + i * num_imu_factors + idx_factor) = bias_acc_j(i);
    }

    Eigen::Matrix<T, 3, 1> bias_gyr_j;
    for (int i = 0; i < bias_gyr_j.size(); i++) {
        bias_gyr_j(i) = *(input_states + 16 * imu_idx_j + 13 + i);
    }
    for (int i = 0; i < bias_gyr_i.size(); i++) {
        *(dev_ptr_set.bias_gyr_j + i * num_imu_factors + idx_factor) = bias_gyr_j(i);
    }

    // ********** end : compute and write outputs **********
}
// instantiation
template __global__ void imu_update<double>(
    int num_imu_factors,
    const double* input_states,
    IMUFactorDevPtrSet<double> dev_ptr_set
);
template __global__ void imu_update<float>(
    int num_imu_factors,
    const float* input_states,
    IMUFactorDevPtrSet<float> dev_ptr_set
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
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_factor >= num_proj_factors)
        return;

    // ********** start : read inputs **********

    int idx_i = *(dev_ptr_set.idx_i + idx_factor);
    int idx_j = *(dev_ptr_set.idx_j + idx_factor);

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    int inv_depth_idx = *(dev_ptr_set.inv_depth_idx + idx_factor);
    *(dev_ptr_set.inv_depth + idx_factor) = *(input_inv_depth + inv_depth_idx);

    Eigen::Matrix<T, 3, 1> p_i;
    for (int i = 0; i < p_i.size(); i++) {
        p_i(i) = *(input_states + 16 * idx_i + 0 + i);
    }
    for (int i = 0; i < p_i.size(); i++) {
        *(dev_ptr_set.p_i + i * num_proj_factors + idx_factor) = p_i(i);
    }

    Eigen::Matrix<T, 3, 1> p_j;
    for (int i = 0; i < p_j.size(); i++) {
        p_j(i) = *(input_states + 16 * idx_j + 0 + i);
    }
    for (int i = 0; i < p_j.size(); i++) {
        *(dev_ptr_set.p_j + i * num_proj_factors + idx_factor) = p_j(i);
    }

    Eigen::Quaternion<T> q_i;
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(input_states + 16 * idx_i + 3 + i);
    }
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        *(dev_ptr_set.q_i + i * num_proj_factors + idx_factor) = q_i.coeffs()(i);
    }

    Eigen::Quaternion<T> q_j;
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(input_states + 16 * idx_j + 3 + i);
    }
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        *(dev_ptr_set.q_j + i * num_proj_factors + idx_factor) = q_j.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> p_ex_0;
    for (int i = 0; i < p_ex_0.size(); i++) {
        p_ex_0(i) = *(input_ex_para_0 + 0 + i);
    }
    for (int i = 0; i < p_ex_0.size(); i++) {
        *(dev_ptr_set.p_ex_0 + i * num_proj_factors + idx_factor) = p_ex_0(i);
    }

    Eigen::Quaternion<T> q_ex_0;
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        q_ex_0.coeffs()(i) = *(input_ex_para_0 + 3 + i);
    }
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        *(dev_ptr_set.q_ex_0 + i * num_proj_factors + idx_factor) = q_ex_0.coeffs()(i);
    }

    *(dev_ptr_set.cur_td + idx_factor) = (*input_cur_td);

    // ********** end : compute and write outputs **********
}
// instantiation
template __global__ void proj_2f1c_update<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_states,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj2F1CFactorDevPtrSet<double> dev_ptr_set
);
template __global__ void proj_2f1c_update<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_states,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj2F1CFactorDevPtrSet<float> dev_ptr_set
);

// ----------

template<typename T>
__global__ void proj_2f2c_update(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F2CFactorDevPtrSet<T> dev_ptr_set
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_factor >= num_proj_factors)
        return;

    // ********** start : read inputs **********

    int idx_i = *(dev_ptr_set.idx_i + idx_factor);
    int idx_j = *(dev_ptr_set.idx_j + idx_factor);

    // ********** end : read inputs **********

    // ********** start : compute and write outputs **********

    int inv_depth_idx = *(dev_ptr_set.inv_depth_idx + idx_factor);
    *(dev_ptr_set.inv_depth + idx_factor) = *(input_inv_depth + inv_depth_idx);

    Eigen::Matrix<T, 3, 1> p_i;
    for (int i = 0; i < p_i.size(); i++) {
        p_i(i) = *(input_states + 16 * idx_i + 0 + i);
    }
    for (int i = 0; i < p_i.size(); i++) {
        *(dev_ptr_set.p_i + i * num_proj_factors + idx_factor) = p_i(i);
    }

    Eigen::Matrix<T, 3, 1> p_j;
    for (int i = 0; i < p_j.size(); i++) {
        p_j(i) = *(input_states + 16 * idx_j + 0 + i);
    }
    for (int i = 0; i < p_j.size(); i++) {
        *(dev_ptr_set.p_j + i * num_proj_factors + idx_factor) = p_j(i);
    }

    Eigen::Quaternion<T> q_i;
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        q_i.coeffs()(i) = *(input_states + 16 * idx_i + 3 + i);
    }
    for (int i = 0; i < q_i.coeffs().size(); i++) {
        *(dev_ptr_set.q_i + i * num_proj_factors + idx_factor) = q_i.coeffs()(i);
    }

    Eigen::Quaternion<T> q_j;
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        q_j.coeffs()(i) = *(input_states + 16 * idx_j + 3 + i);
    }
    for (int i = 0; i < q_j.coeffs().size(); i++) {
        *(dev_ptr_set.q_j + i * num_proj_factors + idx_factor) = q_j.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> p_ex_0;
    for (int i = 0; i < p_ex_0.size(); i++) {
        p_ex_0(i) = *(input_ex_para_0 + 0 + i);
    }
    for (int i = 0; i < p_ex_0.size(); i++) {
        *(dev_ptr_set.p_ex_0 + i * num_proj_factors + idx_factor) = p_ex_0(i);
    }

    Eigen::Quaternion<T> q_ex_0;
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        q_ex_0.coeffs()(i) = *(input_ex_para_0 + 3 + i);
    }
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        *(dev_ptr_set.q_ex_0 + i * num_proj_factors + idx_factor) = q_ex_0.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> p_ex_1;
    for (int i = 0; i < p_ex_1.size(); i++) {
        p_ex_1(i) = *(input_ex_para_1 + 0 + i);
    }
    for (int i = 0; i < p_ex_1.size(); i++) {
        *(dev_ptr_set.p_ex_1 + i * num_proj_factors + idx_factor) = p_ex_1(i);
    }

    Eigen::Quaternion<T> q_ex_1;
    for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
        q_ex_1.coeffs()(i) = *(input_ex_para_1 + 3 + i);
    }
    for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
        *(dev_ptr_set.q_ex_1 + i * num_proj_factors + idx_factor) = q_ex_1.coeffs()(i);
    }

    *(dev_ptr_set.cur_td + idx_factor) = (*input_cur_td);

    // ********** end : compute and write outputs **********
}
// instantiation
template __global__ void proj_2f2c_update<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_ex_para_1,
    const double* input_states,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj2F2CFactorDevPtrSet<double> dev_ptr_set
);
template __global__ void proj_2f2c_update<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_ex_para_1,
    const float* input_states,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj2F2CFactorDevPtrSet<float> dev_ptr_set
);

// ----------

template<typename T>
__global__ void proj_1f2c_update(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set
) {
    unsigned int idx_factor = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_factor >= num_proj_factors)
        return;

    // ********** start : compute and write outputs **********

    int inv_depth_idx = *(dev_ptr_set.inv_depth_idx + idx_factor);
    *(dev_ptr_set.inv_depth + idx_factor) = *(input_inv_depth + inv_depth_idx);

    Eigen::Matrix<T, 3, 1> p_ex_0;
    for (int i = 0; i < p_ex_0.size(); i++) {
        p_ex_0(i) = *(input_ex_para_0 + 0 + i);
    }
    for (int i = 0; i < p_ex_0.size(); i++) {
        *(dev_ptr_set.p_ex_0 + i * num_proj_factors + idx_factor) = p_ex_0(i);
    }

    Eigen::Quaternion<T> q_ex_0;
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        q_ex_0.coeffs()(i) = *(input_ex_para_0 + 3 + i);
    }
    for (int i = 0; i < q_ex_0.coeffs().size(); i++) {
        *(dev_ptr_set.q_ex_0 + i * num_proj_factors + idx_factor) = q_ex_0.coeffs()(i);
    }

    Eigen::Matrix<T, 3, 1> p_ex_1;
    for (int i = 0; i < p_ex_1.size(); i++) {
        p_ex_1(i) = *(input_ex_para_1 + 0 + i);
    }
    for (int i = 0; i < p_ex_1.size(); i++) {
        *(dev_ptr_set.p_ex_1 + i * num_proj_factors + idx_factor) = p_ex_1(i);
    }

    Eigen::Quaternion<T> q_ex_1;
    for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
        q_ex_1.coeffs()(i) = *(input_ex_para_1 + 3 + i);
    }
    for (int i = 0; i < q_ex_1.coeffs().size(); i++) {
        *(dev_ptr_set.q_ex_1 + i * num_proj_factors + idx_factor) = q_ex_1.coeffs()(i);
    }

    *(dev_ptr_set.cur_td + idx_factor) = (*input_cur_td);

    // ********** end : compute and write outputs **********
}
// instantiation
template __global__ void proj_1f2c_update<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_ex_para_1,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj1F2CFactorDevPtrSet<double> dev_ptr_set
);
template __global__ void proj_1f2c_update<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_ex_para_1,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj1F2CFactorDevPtrSet<float> dev_ptr_set
);

// ----------

template<typename T>
void StatesAddDeltaKernel(
    int num_key_frames,
    const T* input_delta_ex_para_0,     // length = 6
    const T* input_delta_ex_para_1,     // length = 6
    const T* input_delta_states,        // stride = 15
    const T* input_delta_cur_td,
    T* output_ex_para_0,                // length = 7
    T* output_ex_para_1,                // length = 7
    T* output_states,                   // stride = 16
    T* output_cur_td
) {
    int _num_threads = 1;
    int _num_blocks = num_key_frames / _num_threads;
    if( (num_key_frames % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    states_add_delta<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_key_frames,
        input_delta_ex_para_0,
        input_delta_ex_para_1,
        input_delta_states,
        input_delta_cur_td,
        output_ex_para_0,
        output_ex_para_1,
        output_states,
        output_cur_td
    );
}
// instantiation
template void StatesAddDeltaKernel<double>(
    int num_key_frames,
    const double* input_delta_ex_para_0,    // length = 6
    const double* input_delta_ex_para_1,    // length = 6
    const double* input_delta_states,       // stride = 15
    const double* input_delta_cur_td,
    double* output_ex_para_0,               // length = 7
    double* output_ex_para_1,               // length = 7
    double* output_states,                  // stride = 16
    double* output_cur_td
);
template void StatesAddDeltaKernel<float>(
    int num_key_frames,
    const float* input_delta_ex_para_0,     // length = 6
    const float* input_delta_ex_para_1,     // length = 6
    const float* input_delta_states,        // stride = 15
    const float* input_delta_cur_td,
    float* output_ex_para_0,                // length = 7
    float* output_ex_para_1,                // length = 7
    float* output_states,                   // stride = 16
    float* output_cur_td
);

// ----------

template<typename T>
void InvDepthAddDeltaKernel(
    int num_world_points,
    const T* input_delta_inv_depth,
    T* output_inv_depth
) {
    int _num_threads = 32;
    int _num_blocks = num_world_points / _num_threads;
    if( (num_world_points % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    inv_depth_add_delta<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_world_points,
        input_delta_inv_depth,
        output_inv_depth
    );
}
// instantiation
template void InvDepthAddDeltaKernel<double>(
    int num_world_points,
    const double* input_delta_inv_depth,
    double* output_inv_depth
);
template void InvDepthAddDeltaKernel<float>(
    int num_world_points,
    const float* input_delta_inv_depth,
    float* output_inv_depth
);

// ----------

template<typename T>
void IMUUpdateKernel(
    int num_imu_factors,
    const T* input_states,
    IMUFactorDevPtrSet<T>& dev_ptr_set
) {
    int _num_threads = 1;
    int _num_blocks = num_imu_factors / _num_threads;
    if( (num_imu_factors % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    imu_update<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_imu_factors,
        input_states,
        dev_ptr_set
    );
}
// instantiation
template void IMUUpdateKernel<double>(
    int num_imu_factors,
    const double* input_states,
    IMUFactorDevPtrSet<double>& dev_ptr_set
);
template void IMUUpdateKernel<float>(
    int num_imu_factors,
    const float* input_states,
    IMUFactorDevPtrSet<float>& dev_ptr_set
);

// ----------

template<typename T>
void Proj2F1CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F1CFactorDevPtrSet<T>& dev_ptr_set
) {
    int _num_threads = 32;
    int _num_blocks = num_proj_factors / _num_threads;
    if( (num_proj_factors % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    proj_2f1c_update<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_proj_factors,
        input_ex_para_0,
        input_states,
        input_inv_depth,
        input_cur_td,
        dev_ptr_set
    );
}
// instantiation
template void Proj2F1CUpdateKernel<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_states,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj2F1CFactorDevPtrSet<double>& dev_ptr_set
);
template void Proj2F1CUpdateKernel<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_states,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj2F1CFactorDevPtrSet<float>& dev_ptr_set
);

// ----------

template<typename T>
void Proj2F2CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj2F2CFactorDevPtrSet<T>& dev_ptr_set
) {
    int _num_threads = 32;
    int _num_blocks = num_proj_factors / _num_threads;
    if( (num_proj_factors % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    proj_2f2c_update<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_proj_factors,
        input_ex_para_0,
        input_ex_para_1,
        input_states,
        input_inv_depth,
        input_cur_td,
        dev_ptr_set
    );
}
// instantiation
template void Proj2F2CUpdateKernel<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_ex_para_1,
    const double* input_states,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj2F2CFactorDevPtrSet<double>& dev_ptr_set
);
template void Proj2F2CUpdateKernel<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_ex_para_1,
    const float* input_states,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj2F2CFactorDevPtrSet<float>& dev_ptr_set
);

// ----------

template<typename T>
void Proj1F2CUpdateKernel(
    int num_proj_factors,
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_inv_depth,
    const T* input_cur_td,
    Proj1F2CFactorDevPtrSet<T>& dev_ptr_set
) {
    int _num_threads = 32;
    int _num_blocks = num_proj_factors / _num_threads;
    if( (num_proj_factors % _num_threads) != 0 )
        _num_blocks += 1;
    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads);

    proj_1f2c_update<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_proj_factors,
        input_ex_para_0,
        input_ex_para_1,
        input_inv_depth,
        input_cur_td,
        dev_ptr_set
    );
}
// instantiation
template void Proj1F2CUpdateKernel<double>(
    int num_proj_factors,
    const double* input_ex_para_0,
    const double* input_ex_para_1,
    const double* input_inv_depth,
    const double* input_cur_td,
    Proj1F2CFactorDevPtrSet<double>& dev_ptr_set
);
template void Proj1F2CUpdateKernel<float>(
    int num_proj_factors,
    const float* input_ex_para_0,
    const float* input_ex_para_1,
    const float* input_inv_depth,
    const float* input_cur_td,
    Proj1F2CFactorDevPtrSet<float>& dev_ptr_set
);

} // namespace VINS_FUSION_CUDA_BA