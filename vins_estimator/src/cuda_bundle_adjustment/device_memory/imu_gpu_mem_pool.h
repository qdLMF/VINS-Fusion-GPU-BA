//
// Created by lmf on 23-7-25.
//

#ifndef CUDA_BA_IMU_GPU_MEM_POOL_H
#define CUDA_BA_IMU_GPU_MEM_POOL_H

#include "GPUMatrix.h"
#include "../imu_allocator.h"
#include "cuda_error_check.h"
#include "imu_dev_ptr_set.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T, unsigned int MAXNUMIMUFACTORS>
class IMUGPUMemPool {
public :
    unsigned int num_imu_factors;

public :
    static constexpr unsigned int MaxNumIMUFactors() { return MAXNUMIMUFACTORS; }

public :
    IMUFactorDevPtrSet<T> dev_ptr_set;

public :
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> gravity;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> p_i;
    CUDAMatrix2DStatic<T,     4, MAXNUMIMUFACTORS> q_i;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> v_i;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> bias_acc_i;
    CUDAMatrix2DStatic<T,     4, MAXNUMIMUFACTORS> bias_gyr_i;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> p_j;
    CUDAMatrix2DStatic<T,     4, MAXNUMIMUFACTORS> q_j;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> v_j;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> bias_acc_j;
    CUDAMatrix2DStatic<T,     4, MAXNUMIMUFACTORS> bias_gyr_j;
    CUDAMatrix2DStatic<T,     1, MAXNUMIMUFACTORS> sum_dt;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> linearized_bias_acc;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> linearized_bias_gyr;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> delta_p;
    CUDAMatrix2DStatic<T,     4, MAXNUMIMUFACTORS> delta_q;
    CUDAMatrix2DStatic<T,     3, MAXNUMIMUFACTORS> delta_v;
    CUDAMatrix2DStatic<T,     9, MAXNUMIMUFACTORS> jacobian_p_bias_acc;  // 9 = 3 x 3
    CUDAMatrix2DStatic<T,     9, MAXNUMIMUFACTORS> jacobian_p_bias_gyr;  // 9 = 3 x 3
    CUDAMatrix2DStatic<T,     9, MAXNUMIMUFACTORS> jacobian_q_bias_gyr;  // 9 = 3 x 3
    CUDAMatrix2DStatic<T,     9, MAXNUMIMUFACTORS> jacobian_v_bias_acc;  // 9 = 3 x 3
    CUDAMatrix2DStatic<T,     9, MAXNUMIMUFACTORS> jacobian_v_bias_gyr;  // 9 = 3 x 3
    CUDAMatrix2DStatic<T,   225, MAXNUMIMUFACTORS> covariance;           // 225 = 15 x 15
    CUDAMatrix2DStatic<T,   225, MAXNUMIMUFACTORS> info;                 // 225 = 15 x 15
    CUDAMatrix2DStatic<T,   225, MAXNUMIMUFACTORS> sqrt_info;            // 225 = 15 x 15
    CUDAMatrix2DStatic<int,   1, MAXNUMIMUFACTORS> idx_i;
    CUDAMatrix2DStatic<int,   1, MAXNUMIMUFACTORS> idx_j;
    CUDAMatrix2DStatic<char,  1, MAXNUMIMUFACTORS> involved_in_marg;

    CUDAMatrix2DStatic<char, 1, MAXNUMIMUFACTORS> pose_i_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMIMUFACTORS> speed_bias_i_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMIMUFACTORS> pose_j_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMIMUFACTORS> speed_bias_j_is_fixed;

public :
    CUDAMatrix2DStatic<T, 225, MAXNUMIMUFACTORS> robust_info;          // 225 = 15 x 15
    CUDAMatrix2DStatic<T,   1, MAXNUMIMUFACTORS> robust_chi2;
    // CUDAMatrix2DStatic<T,   1, MAXNUMIMUFACTORS> drho;
    CUDAMatrix2DStatic<T,  15, MAXNUMIMUFACTORS> residual;
    CUDAMatrix2DStatic<T,  15, MAXNUMIMUFACTORS> drho_info_residual;
    CUDAMatrix2DStatic<T,  90, MAXNUMIMUFACTORS> jacobian_0;           //  90 = 15 x 6 , wrt p_i, q_i
    CUDAMatrix2DStatic<T, 135, MAXNUMIMUFACTORS> jacobian_1;           // 135 = 15 x 9 , wrt v_i, bias_acc_i, bias_gyr_i
    CUDAMatrix2DStatic<T,  90, MAXNUMIMUFACTORS> jacobian_2;           //  90 = 15 x 6 , wrt p_j, q_j
    CUDAMatrix2DStatic<T, 135, MAXNUMIMUFACTORS> jacobian_3;           // 135 = 15 x 9 , wrt v_j, bias_acc_j, bias_gyr_j

    // CUDAMatrix2DStatic<T,  36, MAXNUMIMUFACTORS> hessian_00;           //  36 =  6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_01;           //  54 =  6 x 9 , in Hpp
    // CUDAMatrix2DStatic<T,  36, MAXNUMIMUFACTORS> hessian_02;           //  36 =  6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_03;           //  54 =  6 x 9 , in Hpp

    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_10;           //  54 =  9 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  81, MAXNUMIMUFACTORS> hessian_11;           //  81 =  9 x 9 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_12;           //  54 =  9 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  81, MAXNUMIMUFACTORS> hessian_13;           //  81 =  9 x 9 , in Hpp

    // CUDAMatrix2DStatic<T,  36, MAXNUMIMUFACTORS> hessian_20;           //  36 =  6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_21;           //  54 =  6 x 9 , in Hpp
    // CUDAMatrix2DStatic<T,  36, MAXNUMIMUFACTORS> hessian_22;           //  36 =  6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_23;           //  54 =  6 x 9 , in Hpp

    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_30;           //  54 =  9 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  81, MAXNUMIMUFACTORS> hessian_31;           //  81 =  9 x 9 , in Hpp
    // CUDAMatrix2DStatic<T,  54, MAXNUMIMUFACTORS> hessian_32;           //  54 =  9 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  81, MAXNUMIMUFACTORS> hessian_33;           //  81 =  9 x 9 , in Hpp
    // CUDAMatrix2DStatic<T,   6, MAXNUMIMUFACTORS> rhs_0;                //   6 =  6 x 1 , in Bpp
    
    // CUDAMatrix2DStatic<T,   9, MAXNUMIMUFACTORS> rhs_1;                //   9 =  9 x 1 , in Bpp
    // CUDAMatrix2DStatic<T,   6, MAXNUMIMUFACTORS> rhs_2;                //   6 =  6 x 1 , in Bpp
    // CUDAMatrix2DStatic<T,   9, MAXNUMIMUFACTORS> rhs_3;                //   9 =  9 x 1 , in Bpp

public :
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_00_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_00_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_01_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_01_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_02_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_02_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_03_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_03_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_10_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_10_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_11_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_11_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_12_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_12_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_13_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_13_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_20_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_20_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_21_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_21_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_22_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_22_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_23_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_23_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_30_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_30_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_31_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_31_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_32_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_32_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_33_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> hessian_33_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> rhs_0_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> rhs_1_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> rhs_2_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMIMUFACTORS> rhs_3_row_start;

public :
    IMUGPUMemPool();

public :
    bool Alloc(const IMUFactorAllocator<T>& host_array);
};

template<typename T, unsigned int MAXNUMIMUFACTORS>
IMUGPUMemPool<T, MAXNUMIMUFACTORS>::IMUGPUMemPool() : num_imu_factors(0) {
    dev_ptr_set.gravity = gravity.GetDevRawPtr();
    dev_ptr_set.p_i = p_i.GetDevRawPtr();
    dev_ptr_set.q_i = q_i.GetDevRawPtr();
    dev_ptr_set.v_i = v_i.GetDevRawPtr();
    dev_ptr_set.bias_acc_i = bias_acc_i.GetDevRawPtr();
    dev_ptr_set.bias_gyr_i = bias_gyr_i.GetDevRawPtr();
    dev_ptr_set.p_j = p_j.GetDevRawPtr();
    dev_ptr_set.q_j = q_j.GetDevRawPtr();
    dev_ptr_set.v_j = v_j.GetDevRawPtr();
    dev_ptr_set.bias_acc_j = bias_acc_j.GetDevRawPtr();
    dev_ptr_set.bias_gyr_j = bias_gyr_j.GetDevRawPtr();
    dev_ptr_set.sum_dt = sum_dt.GetDevRawPtr();
    dev_ptr_set.linearized_bias_acc = linearized_bias_acc.GetDevRawPtr();
    dev_ptr_set.linearized_bias_gyr = linearized_bias_gyr.GetDevRawPtr();
    dev_ptr_set.delta_p = delta_p.GetDevRawPtr();
    dev_ptr_set.delta_q = delta_q.GetDevRawPtr();
    dev_ptr_set.delta_v = delta_v.GetDevRawPtr();
    dev_ptr_set.jacobian_p_bias_acc = jacobian_p_bias_acc.GetDevRawPtr();
    dev_ptr_set.jacobian_p_bias_gyr = jacobian_p_bias_gyr.GetDevRawPtr();
    dev_ptr_set.jacobian_q_bias_gyr = jacobian_q_bias_gyr.GetDevRawPtr();
    dev_ptr_set.jacobian_v_bias_acc = jacobian_v_bias_acc.GetDevRawPtr();
    dev_ptr_set.jacobian_v_bias_gyr = jacobian_v_bias_gyr.GetDevRawPtr();
    dev_ptr_set.covariance = covariance.GetDevRawPtr();
    dev_ptr_set.info = info.GetDevRawPtr();
    dev_ptr_set.sqrt_info = sqrt_info.GetDevRawPtr();
    dev_ptr_set.idx_i = idx_i.GetDevRawPtr();
    dev_ptr_set.idx_j = idx_j.GetDevRawPtr();
    dev_ptr_set.involved_in_marg = involved_in_marg.GetDevRawPtr();

    dev_ptr_set.pose_i_is_fixed = pose_i_is_fixed.GetDevRawPtr();
    dev_ptr_set.speed_bias_i_is_fixed = speed_bias_i_is_fixed.GetDevRawPtr();
    dev_ptr_set.pose_j_is_fixed = pose_j_is_fixed.GetDevRawPtr();
    dev_ptr_set.speed_bias_j_is_fixed = speed_bias_j_is_fixed.GetDevRawPtr();

    dev_ptr_set.robust_info = robust_info.GetDevRawPtr();
    dev_ptr_set.robust_chi2 = robust_chi2.GetDevRawPtr();
    // dev_ptr_set.drho = drho.GetDevRawPtr();
    dev_ptr_set.residual = residual.GetDevRawPtr();
    dev_ptr_set.drho_info_residual = drho_info_residual.GetDevRawPtr();
    dev_ptr_set.jacobian_0 = jacobian_0.GetDevRawPtr();
    dev_ptr_set.jacobian_1 = jacobian_1.GetDevRawPtr();
    dev_ptr_set.jacobian_2 = jacobian_2.GetDevRawPtr();
    dev_ptr_set.jacobian_3 = jacobian_3.GetDevRawPtr();

    // dev_ptr_set.hessian_00 = hessian_00.GetDevRawPtr();
    // dev_ptr_set.hessian_01 = hessian_01.GetDevRawPtr();
    // dev_ptr_set.hessian_02 = hessian_02.GetDevRawPtr();
    // dev_ptr_set.hessian_03 = hessian_03.GetDevRawPtr();

    // dev_ptr_set.hessian_10 = hessian_10.GetDevRawPtr();
    // dev_ptr_set.hessian_11 = hessian_11.GetDevRawPtr();
    // dev_ptr_set.hessian_12 = hessian_12.GetDevRawPtr();
    // dev_ptr_set.hessian_13 = hessian_13.GetDevRawPtr();

    // dev_ptr_set.hessian_20 = hessian_20.GetDevRawPtr();
    // dev_ptr_set.hessian_21 = hessian_21.GetDevRawPtr();
    // dev_ptr_set.hessian_22 = hessian_22.GetDevRawPtr();
    // dev_ptr_set.hessian_23 = hessian_23.GetDevRawPtr();

    // dev_ptr_set.hessian_30 = hessian_30.GetDevRawPtr();
    // dev_ptr_set.hessian_31 = hessian_31.GetDevRawPtr();
    // dev_ptr_set.hessian_32 = hessian_32.GetDevRawPtr();
    // dev_ptr_set.hessian_33 = hessian_33.GetDevRawPtr();
    
    // dev_ptr_set.rhs_0 = rhs_0.GetDevRawPtr();
    // dev_ptr_set.rhs_1 = rhs_1.GetDevRawPtr();
    // dev_ptr_set.rhs_2 = rhs_2.GetDevRawPtr();
    // dev_ptr_set.rhs_3 = rhs_3.GetDevRawPtr();

    dev_ptr_set.hessian_00_row_start = hessian_00_row_start.GetDevRawPtr(); dev_ptr_set.hessian_00_col_start = hessian_00_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_01_row_start = hessian_01_row_start.GetDevRawPtr(); dev_ptr_set.hessian_01_col_start = hessian_01_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_02_row_start = hessian_02_row_start.GetDevRawPtr(); dev_ptr_set.hessian_02_col_start = hessian_02_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_03_row_start = hessian_03_row_start.GetDevRawPtr(); dev_ptr_set.hessian_03_col_start = hessian_03_col_start.GetDevRawPtr();

    dev_ptr_set.hessian_10_row_start = hessian_10_row_start.GetDevRawPtr(); dev_ptr_set.hessian_10_col_start = hessian_10_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_11_row_start = hessian_11_row_start.GetDevRawPtr(); dev_ptr_set.hessian_11_col_start = hessian_11_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_12_row_start = hessian_12_row_start.GetDevRawPtr(); dev_ptr_set.hessian_12_col_start = hessian_12_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_13_row_start = hessian_13_row_start.GetDevRawPtr(); dev_ptr_set.hessian_13_col_start = hessian_13_col_start.GetDevRawPtr();

    dev_ptr_set.hessian_20_row_start = hessian_20_row_start.GetDevRawPtr(); dev_ptr_set.hessian_20_col_start = hessian_20_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_21_row_start = hessian_21_row_start.GetDevRawPtr(); dev_ptr_set.hessian_21_col_start = hessian_21_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_22_row_start = hessian_22_row_start.GetDevRawPtr(); dev_ptr_set.hessian_22_col_start = hessian_22_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_23_row_start = hessian_23_row_start.GetDevRawPtr(); dev_ptr_set.hessian_23_col_start = hessian_23_col_start.GetDevRawPtr();

    dev_ptr_set.hessian_30_row_start = hessian_30_row_start.GetDevRawPtr(); dev_ptr_set.hessian_30_col_start = hessian_30_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_31_row_start = hessian_31_row_start.GetDevRawPtr(); dev_ptr_set.hessian_31_col_start = hessian_31_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_32_row_start = hessian_32_row_start.GetDevRawPtr(); dev_ptr_set.hessian_32_col_start = hessian_32_col_start.GetDevRawPtr();
    dev_ptr_set.hessian_33_row_start = hessian_33_row_start.GetDevRawPtr(); dev_ptr_set.hessian_33_col_start = hessian_33_col_start.GetDevRawPtr();

    dev_ptr_set.rhs_0_row_start = rhs_0_row_start.GetDevRawPtr();
    dev_ptr_set.rhs_1_row_start = rhs_1_row_start.GetDevRawPtr();
    dev_ptr_set.rhs_2_row_start = rhs_2_row_start.GetDevRawPtr();
    dev_ptr_set.rhs_3_row_start = rhs_3_row_start.GetDevRawPtr();
}

template<typename T, unsigned int MAXNUMIMUFACTORS>
bool IMUGPUMemPool<T, MAXNUMIMUFACTORS>::Alloc(const IMUFactorAllocator<T>& host_array) {
    if(host_array.num_factors > MaxNumIMUFactors()) {
        return false;
    }

    num_imu_factors = host_array.num_factors;

    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.gravity,
            host_array.gravity_v2.data(),
            sizeof(T) * host_array.gravity_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : gravity")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.p_i,
            host_array.p_i_v2.data(),
            sizeof(T) * host_array.p_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : p_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.q_i,
            host_array.q_i_v2.data(),
            sizeof(T) * host_array.q_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : q_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.v_i,
            host_array.v_i_v2.data(),
            sizeof(T) * host_array.v_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : v_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.bias_acc_i,
            host_array.bias_acc_i_v2.data(),
            sizeof(T) * host_array.bias_acc_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : bias_acc_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.bias_gyr_i,
            host_array.bias_gyr_i_v2.data(),
            sizeof(T) * host_array.bias_gyr_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : bias_gyr_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.p_j,
            host_array.p_j_v2.data(),
            sizeof(T) * host_array.p_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : p_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.q_j,
            host_array.q_j_v2.data(),
            sizeof(T) * host_array.q_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : q_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
            cudaMemcpy(
                    dev_ptr_set.v_j,
                    host_array.v_j_v2.data(),
                    sizeof(T) * host_array.v_j_v2.size(),
                    cudaMemcpyHostToDevice
            ),
            std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : v_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.bias_acc_j,
            host_array.bias_acc_j_v2.data(),
            sizeof(T) * host_array.bias_acc_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : bias_acc_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.bias_gyr_j,
            host_array.bias_gyr_j_v2.data(),
            sizeof(T) * host_array.bias_gyr_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : bias_gyr_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.sum_dt,
            host_array.sum_dt_v2.data(),
            sizeof(T) * host_array.sum_dt_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : sum_dt")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.linearized_bias_acc,
            host_array.linearized_bias_acc_v2.data(),
            sizeof(T) * host_array.linearized_bias_acc_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : linearized_bias_acc")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.linearized_bias_gyr,
            host_array.linearized_bias_gyr_v2.data(),
            sizeof(T) * host_array.linearized_bias_gyr_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : linearized_bias_gyr")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.delta_p,
            host_array.delta_p_v2.data(),
            sizeof(T) * host_array.delta_p_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : delta_p")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.delta_q,
            host_array.delta_q_v2.data(),
            sizeof(T) * host_array.delta_q_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : delta_q")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.delta_v,
            host_array.delta_v_v2.data(),
            sizeof(T) * host_array.delta_v_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : delta_v")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.jacobian_p_bias_acc,
            host_array.jacobian_p_bias_acc_v2.data(),
            sizeof(T) * host_array.jacobian_p_bias_acc_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : jacobian_p_bias_acc")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.jacobian_p_bias_gyr,
            host_array.jacobian_p_bias_gyr_v2.data(),
            sizeof(T) * host_array.jacobian_p_bias_gyr_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : jacobian_p_bias_gyr")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.jacobian_q_bias_gyr,
            host_array.jacobian_q_bias_gyr_v2.data(),
            sizeof(T) * host_array.jacobian_q_bias_gyr_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : jacobian_q_bias_gyr")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.jacobian_v_bias_acc,
            host_array.jacobian_v_bias_acc_v2.data(),
            sizeof(T) * host_array.jacobian_v_bias_acc_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : jacobian_v_bias_acc")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.jacobian_v_bias_gyr,
            host_array.jacobian_v_bias_gyr_v2.data(),
            sizeof(T) * host_array.jacobian_v_bias_gyr_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : jacobian_v_bias_gyr")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.covariance,
            host_array.state_covariance_v2.data(),
            sizeof(T) * host_array.state_covariance_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : covariance")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.info,
            host_array.info_v2.data(),
            sizeof(T) * host_array.info_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : info")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.sqrt_info,
            host_array.sqrt_info_v2.data(),
            sizeof(T) * host_array.sqrt_info_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : sqrt_info")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.idx_i,
            host_array.imu_idx_i.data(),
            sizeof(int) * host_array.imu_idx_i.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : idx_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.idx_j,
            host_array.imu_idx_j.data(),
            sizeof(int) * host_array.imu_idx_j.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : idx_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.involved_in_marg,
            host_array.involved_in_marg.data(),
            sizeof(char) * host_array.involved_in_marg.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : involved_in_marg")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pose_i_is_fixed,
            host_array.pose_i_is_fixed.data(),
            sizeof(char) * host_array.pose_i_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : pose_i_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.speed_bias_i_is_fixed,
            host_array.speed_bias_i_is_fixed.data(),
            sizeof(char) * host_array.speed_bias_i_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : speed_bias_i_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pose_j_is_fixed,
            host_array.pose_j_is_fixed.data(),
            sizeof(char) * host_array.pose_j_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : pose_j_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.speed_bias_j_is_fixed,
            host_array.speed_bias_j_is_fixed.data(),
            sizeof(char) * host_array.speed_bias_j_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in IMUGPUMemPool::Alloc() : speed_bias_j_is_fixed")
    );

    return true;
}

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_IMU_GPU_MEM_POOL_H
