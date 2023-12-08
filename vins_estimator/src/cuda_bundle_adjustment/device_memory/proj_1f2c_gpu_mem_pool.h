//
// Created by lmf on 23-7-25.
//

#ifndef CUDA_BA_PROJ_1F2C_GPU_MEM_POOL_H
#define CUDA_BA_PROJ_1F2C_GPU_MEM_POOL_H

#include "GPUMatrix.h"
#include "../proj_allocators.h"
#include "cuda_error_check.h"
#include "proj_1f2c_dev_ptr_set.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T, unsigned int MAXNUMPROJFACTORS>
class Proj1F2CGPUMemPool {
public :
    unsigned int num_proj_factors;

public :
    static constexpr unsigned int MaxNumProjFactors() { return MAXNUMPROJFACTORS; }

public :
    Proj1F2CFactorDevPtrSet<T> dev_ptr_set;

public :
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_i;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_j;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> velocity_i;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> velocity_j;
    CUDAMatrix2DStatic<T, 1, MAXNUMPROJFACTORS> td_i;
    CUDAMatrix2DStatic<T, 1, MAXNUMPROJFACTORS> td_j;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> p_ex_0;
    CUDAMatrix2DStatic<T, 4, MAXNUMPROJFACTORS> q_ex_0;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> p_ex_1;
    CUDAMatrix2DStatic<T, 4, MAXNUMPROJFACTORS> q_ex_1;
    CUDAMatrix2DStatic<T, 1, MAXNUMPROJFACTORS> inv_depth;
    CUDAMatrix2DStatic<T, 1, MAXNUMPROJFACTORS> cur_td;
    CUDAMatrix2DStatic<int,  1, MAXNUMPROJFACTORS> idx_i;
    CUDAMatrix2DStatic<int,  1, MAXNUMPROJFACTORS> idx_j;
    CUDAMatrix2DStatic<int,  1, MAXNUMPROJFACTORS> inv_depth_idx;
    CUDAMatrix2DStatic<int,  1, MAXNUMPROJFACTORS> inv_depth_idx_for_marg;
    CUDAMatrix2DStatic<char, 1, MAXNUMPROJFACTORS> involved_in_marg;

    CUDAMatrix2DStatic<char, 1, MAXNUMPROJFACTORS> pose_ex_0_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMPROJFACTORS> pose_ex_1_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMPROJFACTORS> inv_depth_is_fixed;
    CUDAMatrix2DStatic<char, 1, MAXNUMPROJFACTORS> cur_td_is_fixed;

public :
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_i_td;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_j_td;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_cam_i;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_imu_i;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_imu_j;
    CUDAMatrix2DStatic<T, 3, MAXNUMPROJFACTORS> pts_cam_j;
    CUDAMatrix2DStatic<T, 6, MAXNUMPROJFACTORS> reduce;         // 6 = 2 x 3
    CUDAMatrix2DStatic<T, 9, MAXNUMPROJFACTORS> r_ex_0;         // 9 = 3 x 3
    CUDAMatrix2DStatic<T, 9, MAXNUMPROJFACTORS> r_ex_0_trans;   // 9 = 3 x 3
    CUDAMatrix2DStatic<T, 9, MAXNUMPROJFACTORS> r_ex_1;         // 9 = 3 x 3
    CUDAMatrix2DStatic<T, 9, MAXNUMPROJFACTORS> r_ex_1_trans;   // 9 = 3 x 3
    CUDAMatrix2DStatic<T, 9, MAXNUMPROJFACTORS> tmp_r_1;        // 9 = 3 x 3

public :
    CUDAMatrix2DStatic<T,  4, MAXNUMPROJFACTORS> robust_info;          // 4 = 2 x 2
    CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> robust_chi2;          // 1
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> drho;                 // 1 = 1 x 1
    CUDAMatrix2DStatic<T,  2, MAXNUMPROJFACTORS> residual;             // 2
    CUDAMatrix2DStatic<T,  2, MAXNUMPROJFACTORS> drho_info_residual;   // 2

    CUDAMatrix2DStatic<T, 12, MAXNUMPROJFACTORS> jacobian_0;           // 12 = 2 x 6 , wrt p_ex_0, q_ex_0
    CUDAMatrix2DStatic<T, 12, MAXNUMPROJFACTORS> jacobian_1;           // 12 = 2 x 6 , wrt p_ex_1, q_ex_1
    CUDAMatrix2DStatic<T,  2, MAXNUMPROJFACTORS> jacobian_2;           //  2 = 2 x 1 , wrt inv_depth
    CUDAMatrix2DStatic<T,  2, MAXNUMPROJFACTORS> jacobian_3;           //  2 = 2 x 1 , wrt cur_td

    // CUDAMatrix2DStatic<T, 36, MAXNUMPROJFACTORS> hessian_00;           // 36 = 6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T, 36, MAXNUMPROJFACTORS> hessian_01;           // 36 = 6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_02;           //  6 = 6 x 1 , in Hpm
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_03;           //  6 = 6 x 1 , in Hpp

    // CUDAMatrix2DStatic<T, 36, MAXNUMPROJFACTORS> hessian_10;           // 36 = 6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T, 36, MAXNUMPROJFACTORS> hessian_11;           // 36 = 6 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_12;           //  6 = 6 x 1 , in Hpm
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_13;           //  6 = 6 x 1 , in Hpp

    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_20;           //  6 = 1 x 6 , in Hmp
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_21;           //  6 = 1 x 6 , in Hmp
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> hessian_22;           //  1 = 1 x 1 , in Hmm
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> hessian_23;           //  1 = 1 x 1 , in Hmp

    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_30;           //  6 = 1 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> hessian_31;           //  6 = 1 x 6 , in Hpp
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> hessian_32;           //  1 = 1 x 1 , in Hpm
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> hessian_33;           //  1 = 1 x 1 , in Hpp

    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> rhs_0;                //  6 = 6 x 1 , in Bpp
    // CUDAMatrix2DStatic<T,  6, MAXNUMPROJFACTORS> rhs_1;                //  6 = 6 x 1 , in Bpp
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> rhs_2;                //  1 = 1 x 1 , in Bmm
    // CUDAMatrix2DStatic<T,  1, MAXNUMPROJFACTORS> rhs_3;                //  1 = 1 x 1 , in Bpp

public :
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_00_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_00_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_01_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_01_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_02_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_02_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_03_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_03_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_10_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_10_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_11_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_11_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_12_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_12_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_13_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_13_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_20_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_20_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_21_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_21_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_22_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_22_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_23_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_23_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_30_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_30_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_31_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_31_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_32_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_32_col_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_33_row_start; CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> hessian_33_col_start;

    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> rhs_0_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> rhs_1_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> rhs_2_row_start;
    CUDAMatrix2DStatic<unsigned int, 1, MAXNUMPROJFACTORS> rhs_3_row_start;

public :
    Proj1F2CGPUMemPool();

public :
    bool Alloc(const Proj1F2CFactorAllocator<T>& host_array);
};

template<typename T, unsigned int MAXNUMPROJFACTORS>
Proj1F2CGPUMemPool<T, MAXNUMPROJFACTORS>::Proj1F2CGPUMemPool() : num_proj_factors(0) {
    dev_ptr_set.pts_i = pts_i.GetDevRawPtr();
    dev_ptr_set.pts_j = pts_j.GetDevRawPtr();
    dev_ptr_set.velocity_i = velocity_i.GetDevRawPtr();
    dev_ptr_set.velocity_j = velocity_j.GetDevRawPtr();
    dev_ptr_set.td_i = td_i.GetDevRawPtr();
    dev_ptr_set.td_j = td_j.GetDevRawPtr();
    dev_ptr_set.p_ex_0 = p_ex_0.GetDevRawPtr();
    dev_ptr_set.q_ex_0 = q_ex_0.GetDevRawPtr();
    dev_ptr_set.p_ex_1 = p_ex_1.GetDevRawPtr();
    dev_ptr_set.q_ex_1 = q_ex_1.GetDevRawPtr();
    dev_ptr_set.inv_depth = inv_depth.GetDevRawPtr();
    dev_ptr_set.cur_td = cur_td.GetDevRawPtr();
    dev_ptr_set.idx_i = idx_i.GetDevRawPtr();
    dev_ptr_set.idx_j = idx_j.GetDevRawPtr();
    dev_ptr_set.inv_depth_idx = inv_depth_idx.GetDevRawPtr();
    dev_ptr_set.inv_depth_idx_for_marg = inv_depth_idx_for_marg.GetDevRawPtr();
    dev_ptr_set.involved_in_marg = involved_in_marg.GetDevRawPtr();

    dev_ptr_set.pose_ex_0_is_fixed = pose_ex_0_is_fixed.GetDevRawPtr();
    dev_ptr_set.pose_ex_1_is_fixed = pose_ex_1_is_fixed.GetDevRawPtr();
    dev_ptr_set.inv_depth_is_fixed = inv_depth_is_fixed.GetDevRawPtr();
    dev_ptr_set.cur_td_is_fixed = cur_td_is_fixed.GetDevRawPtr();

    dev_ptr_set.pts_i_td = pts_i_td.GetDevRawPtr();
    dev_ptr_set.pts_j_td = pts_j_td.GetDevRawPtr();
    dev_ptr_set.pts_cam_i = pts_cam_i.GetDevRawPtr();
    dev_ptr_set.pts_imu_i = pts_imu_i.GetDevRawPtr();
    dev_ptr_set.pts_imu_j = pts_imu_j.GetDevRawPtr();
    dev_ptr_set.pts_cam_j = pts_cam_j.GetDevRawPtr();
    dev_ptr_set.reduce = reduce.GetDevRawPtr();
    dev_ptr_set.r_ex_0 = r_ex_0.GetDevRawPtr();
    dev_ptr_set.r_ex_0_trans = r_ex_0_trans.GetDevRawPtr();
    dev_ptr_set.r_ex_1 = r_ex_1.GetDevRawPtr();
    dev_ptr_set.r_ex_1_trans = r_ex_1_trans.GetDevRawPtr();
    dev_ptr_set.tmp_r_1 = tmp_r_1.GetDevRawPtr();

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

// ------------------------------------------------------------------------------------------------------------------------

template<typename T, unsigned int MAXNUMPROJFACTORS>
bool Proj1F2CGPUMemPool<T, MAXNUMPROJFACTORS>::Alloc(const Proj1F2CFactorAllocator<T>& host_array) {
    if(host_array.num_factors > MaxNumProjFactors()) {
        return false;
    }

    num_proj_factors = host_array.num_factors;

    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pts_i,
            host_array.pts_i_v2.data(),
            sizeof(T) * host_array.pts_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : pts_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pts_j,
            host_array.pts_j_v2.data(),
            sizeof(T) * host_array.pts_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : pts_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.velocity_i,
            host_array.velocity_i_v2.data(),
            sizeof(T) * host_array.velocity_i_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : velocity_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.velocity_j,
            host_array.velocity_j_v2.data(),
            sizeof(T) * host_array.velocity_j_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : velocity_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.td_i,
            host_array.td_i.data(),
            sizeof(T) * host_array.td_i.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : td_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.td_j,
            host_array.td_j.data(),
            sizeof(T) * host_array.td_j.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : td_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.p_ex_0,
            host_array.p_ex_0_v2.data(),
            sizeof(T) * host_array.p_ex_0_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : p_ex_0")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.q_ex_0,
            host_array.q_ex_0_v2.data(),
            sizeof(T) * host_array.q_ex_0_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : q_ex_0")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.p_ex_1,
            host_array.p_ex_1_v2.data(),
            sizeof(T) * host_array.p_ex_1_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : p_ex_1")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.q_ex_1,
            host_array.q_ex_1_v2.data(),
            sizeof(T) * host_array.q_ex_1_v2.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : q_ex_1")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.inv_depth,
            host_array.inv_depth.data(),
            sizeof(T) * host_array.inv_depth.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : inv_depth")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.cur_td,
            host_array.time_delay.data(),
            sizeof(T) * host_array.time_delay.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : cur_td")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.idx_i,
            host_array.imu_idx_i.data(),
            sizeof(int) * host_array.imu_idx_i.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : idx_i")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.idx_j,
            host_array.imu_idx_j.data(),
            sizeof(int) * host_array.imu_idx_j.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : idx_j")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.inv_depth_idx,
            host_array.inv_depth_idx.data(),
            sizeof(int) * host_array.inv_depth_idx.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : inv_depth_idx")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.inv_depth_idx_for_marg,
            host_array.inv_depth_idx_for_marg.data(),
            sizeof(int) * host_array.inv_depth_idx_for_marg.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : inv_depth_idx_for_marg")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.involved_in_marg,
            host_array.involved_in_marg.data(),
            sizeof(char) * host_array.involved_in_marg.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : involved_in_marg")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pose_ex_0_is_fixed,
            host_array.pose_ex_0_is_fixed.data(),
            sizeof(char) * host_array.pose_ex_0_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : pose_ex_0_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.pose_ex_1_is_fixed,
            host_array.pose_ex_1_is_fixed.data(),
            sizeof(char) * host_array.pose_ex_1_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : pose_ex_1_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.inv_depth_is_fixed,
            host_array.inv_depth_is_fixed.data(),
            sizeof(char) * host_array.inv_depth_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : inv_depth_is_fixed")
    );
    RETURN_FALSE_IF_CUDA_ERROR(
        cudaMemcpy(
            dev_ptr_set.cur_td_is_fixed,
            host_array.cur_td_is_fixed.data(),
            sizeof(char) * host_array.cur_td_is_fixed.size(),
            cudaMemcpyHostToDevice
        ),
        std::string("ALLOC ERROR in Proj1F2CGPUMemPool::Alloc() : cur_td_is_fixed")
    );

    return true;
}

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_PROJ_1F2C_GPU_MEM_POOL_H
