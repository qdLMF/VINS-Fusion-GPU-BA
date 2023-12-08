//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_IMU_DEV_PTR_SET_H
#define CUDA_BA_IMU_DEV_PTR_SET_H

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class IMUFactorDevPtrSet {
public :
    // input
    T* gravity = nullptr;               // inner dim * outter dim : num_imu_factors * 3
    T* p_i = nullptr;                   // inner dim * outter dim : num_imu_factors * 3
    T* q_i = nullptr;                   // inner dim * outter dim : num_imu_factors * 4
    T* v_i = nullptr;                   // inner dim * outter dim : num_imu_factors * 3
    T* bias_acc_i = nullptr;            // inner dim * outter dim : num_imu_factors * 3
    T* bias_gyr_i = nullptr;            // inner dim * outter dim : num_imu_factors * 3
    T* p_j = nullptr;                   // inner dim * outter dim : num_imu_factors * 3
    T* q_j = nullptr;                   // inner dim * outter dim : num_imu_factors * 4
    T* v_j = nullptr;                   // inner dim * outter dim : num_imu_factors * 3
    T* bias_acc_j = nullptr;            // inner dim * outter dim : num_imu_factors * 3
    T* bias_gyr_j = nullptr;            // inner dim * outter dim : num_imu_factors * 3
    T* sum_dt = nullptr;                // inner dim * outter dim : num_imu_factors * 1
    T* linearized_bias_acc = nullptr;   // inner dim * outter dim : num_imu_factors * 3
    T* linearized_bias_gyr = nullptr;   // inner dim * outter dim : num_imu_factors * 3
    T* delta_p = nullptr;               // inner dim * outter dim : num_imu_factors * 3
    T* delta_q = nullptr;               // inner dim * outter dim : num_imu_factors * 4
    T* delta_v = nullptr;               // inner dim * outter dim : num_imu_factors * 3
    T* jacobian_p_bias_acc = nullptr;   // inner dim * outter dim : num_imu_factors * 9 = 3 x 3
    T* jacobian_p_bias_gyr = nullptr;   // inner dim * outter dim : num_imu_factors * 9 = 3 x 3
    T* jacobian_q_bias_gyr = nullptr;   // inner dim * outter dim : num_imu_factors * 9 = 3 x 3
    T* jacobian_v_bias_acc = nullptr;   // inner dim * outter dim : num_imu_factors * 9 = 3 x 3
    T* jacobian_v_bias_gyr = nullptr;   // inner dim * outter dim : num_imu_factors * 9 = 3 x 3
    T* covariance = nullptr;            // inner dim * outter dim : num_imu_factors * 225 = 15 x 15
    T* info = nullptr;                  // inner dim * outter dim : num_imu_factors * 225 = 15 x 15
    T* sqrt_info = nullptr;             // inner dim * outter dim : num_imu_factors * 225 = 15 x 15
    int* idx_i = nullptr;               // inner dim * outter dim : num_imu_factors * 1
    int* idx_j = nullptr;               // inner dim * outter dim : num_imu_factors * 1
    char* involved_in_marg = nullptr;   // inner dim * outter dim : num_imu_factors * 1

    char* pose_i_is_fixed = nullptr;            // inner dim * outter dim : num_projection_factors * 1
    char* speed_bias_i_is_fixed = nullptr;      // inner dim * outter dim : num_projection_factors * 1
    char* pose_j_is_fixed = nullptr;            // inner dim * outter dim : num_projection_factors * 1
    char* speed_bias_j_is_fixed = nullptr;      // inner dim * outter dim : num_projection_factors * 1

    // output
    T* robust_info = nullptr;           // inner dim * outter dim : num_imu_factors * 225 = 15 x 15
    T* robust_chi2 = nullptr;           // inner dim * outter dim : num_imu_factors * 1
    // T* drho = nullptr;                  // inner dim * outter dim : num_imu_factors * 1
    T* residual = nullptr;              // inner dim * outter dim : num_imu_factors * 15
    T* drho_info_residual = nullptr;    // num_imu_factors * 15
    T* jacobian_0 = nullptr;            // num_imu_factors *  90 = 15 x 6 , wrt p_i, q_i
    T* jacobian_1 = nullptr;            // num_imu_factors * 135 = 15 x 9 , wrt v_i, bias_acc_i, bias_gyr_i
    T* jacobian_2 = nullptr;            // num_imu_factors *  90 = 15 x 6 , wrt p_j, q_j
    T* jacobian_3 = nullptr;            // num_imu_factors * 135 = 15 x 9 , wrt v_j, bias_acc_j, bias_gyr_j
    // T* hessian_00 = nullptr;            // num_imu_factors *  36 =  6 x 6 , in Hpp
    // T* hessian_01 = nullptr;            // num_imu_factors *  54 =  6 x 9 , in Hpp
    // T* hessian_02 = nullptr;            // num_imu_factors *  36 =  6 x 6 , in Hpp
    // T* hessian_03 = nullptr;            // num_imu_factors *  54 =  6 x 9 , in Hpp
    // T* hessian_10 = nullptr;            // num_imu_factors *  54 =  9 x 6 , in Hpp
    // T* hessian_11 = nullptr;            // num_imu_factors *  81 =  9 x 9 , in Hpp
    // T* hessian_12 = nullptr;            // num_imu_factors *  54 =  9 x 6 , in Hpp
    // T* hessian_13 = nullptr;            // num_imu_factors *  81 =  9 x 9 , in Hpp
    // T* hessian_20 = nullptr;            // num_imu_factors *  36 =  6 x 6 , in Hpp
    // T* hessian_21 = nullptr;            // num_imu_factors *  54 =  6 x 9 , in Hpp
    // T* hessian_22 = nullptr;            // num_imu_factors *  36 =  6 x 6 , in Hpp
    // T* hessian_23 = nullptr;            // num_imu_factors *  54 =  6 x 9 , in Hpp
    // T* hessian_30 = nullptr;            // num_imu_factors *  54 =  9 x 6 , in Hpp
    // T* hessian_31 = nullptr;            // num_imu_factors *  81 =  9 x 9 , in Hpp
    // T* hessian_32 = nullptr;            // num_imu_factors *  54 =  9 x 6 , in Hpp
    // T* hessian_33 = nullptr;            // num_imu_factors *  81 =  9 x 9 , in Hpp
    // T* rhs_0 = nullptr;                 // num_imu_factors *   6 =  6 x 1 , in Bpp
    // T* rhs_1 = nullptr;                 // num_imu_factors *   9 =  9 x 1 , in Bpp
    // T* rhs_2 = nullptr;                 // num_imu_factors *   6 =  6 x 1 , in Bpp
    // T* rhs_3 = nullptr;                 // num_imu_factors *   9 =  6 x 1 , in Bpp

    unsigned int* hessian_00_row_start = nullptr; unsigned int* hessian_00_col_start = nullptr;
    unsigned int* hessian_01_row_start = nullptr; unsigned int* hessian_01_col_start = nullptr;
    unsigned int* hessian_02_row_start = nullptr; unsigned int* hessian_02_col_start = nullptr;
    unsigned int* hessian_03_row_start = nullptr; unsigned int* hessian_03_col_start = nullptr;

    unsigned int* hessian_10_row_start = nullptr; unsigned int* hessian_10_col_start = nullptr;
    unsigned int* hessian_11_row_start = nullptr; unsigned int* hessian_11_col_start = nullptr;
    unsigned int* hessian_12_row_start = nullptr; unsigned int* hessian_12_col_start = nullptr;
    unsigned int* hessian_13_row_start = nullptr; unsigned int* hessian_13_col_start = nullptr;

    unsigned int* hessian_20_row_start = nullptr; unsigned int* hessian_20_col_start = nullptr;
    unsigned int* hessian_21_row_start = nullptr; unsigned int* hessian_21_col_start = nullptr;
    unsigned int* hessian_22_row_start = nullptr; unsigned int* hessian_22_col_start = nullptr;
    unsigned int* hessian_23_row_start = nullptr; unsigned int* hessian_23_col_start = nullptr;

    unsigned int* hessian_30_row_start = nullptr; unsigned int* hessian_30_col_start = nullptr;
    unsigned int* hessian_31_row_start = nullptr; unsigned int* hessian_31_col_start = nullptr;
    unsigned int* hessian_32_row_start = nullptr; unsigned int* hessian_32_col_start = nullptr;
    unsigned int* hessian_33_row_start = nullptr; unsigned int* hessian_33_col_start = nullptr;

    unsigned int* rhs_0_row_start = nullptr;
    unsigned int* rhs_1_row_start = nullptr;
    unsigned int* rhs_2_row_start = nullptr;
    unsigned int* rhs_3_row_start = nullptr;

public :
    void SetNullptr();
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_IMU_DEV_PTR_SET_H
