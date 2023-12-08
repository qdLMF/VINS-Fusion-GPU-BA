//
// Created by lmf on 23-8-1.
//

#ifndef CUDA_BA_SOLVE_DELTA_CUH
#define CUDA_BA_SOLVE_DELTA_CUH

#include <cstdio>
#include <iostream>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void SolveDeltaKernels(
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp,
    int num_cols_Hpp,
    T* dev_ptr_Hpp,
    int num_elem_Hmm_diag,
    T* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    T* dev_ptr_Hpm,
    int num_rows_Hmp,
    int num_cols_Hmp,
    T* dev_ptr_Hmp,
    int num_elem_Bpp,
    T* dev_ptr_Bpp,
    int num_elem_Bmm,
    T* dev_ptr_Bmm,
    T* dev_ptr_lambda,
    T* dev_ptr_Hpp_with_lambda,
    T* dev_ptr_Hmm_diag_with_lambda,
    T* dev_ptr_HpmHmmInv,
    T* dev_ptr_Hpp_schur,
    T* dev_ptr_Bpp_schur,
    T* dev_ptr_delta_Xpp,
    T* dev_ptr_delta_inv_depth
);

template<typename T>
void MargInvDepthKernels(
    cublasHandle_t& cublas_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    T* dev_ptr_Hpp_for_marg,
    int num_elem_Hmm_diag_for_marg,
    T* dev_ptr_Hmm_diag_for_marg,
    int num_rows_Hpm_for_marg,
    int num_cols_Hpm_for_marg,
    T* dev_ptr_Hpm_for_marg,
    int num_rows_Hmp_for_marg,
    int num_cols_Hmp_for_marg,
    T* dev_ptr_Hmp_for_marg,
    int num_elem_Bpp_for_marg,
    T* dev_ptr_Bpp_for_marg,
    int num_elem_Bmm_for_marg,
    T* dev_ptr_Bmm_for_marg
);

template<typename T>
void MargKeyFrameKernelsWithPriorDecomp(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    T* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    T* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    T* dev_ptr_Hpp_prior,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    T* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    T* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    T* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    T* dev_ptr_H22,
    int num_elem_B11,
    T* dev_ptr_B11,
    int num_elem_B22,
    T* dev_ptr_B22,
    T* dev_ptr_Hprior_eigenvec,
    T* dev_ptr_Hprior_eigenval
);

template<typename T>
void MargKeyFrameKernelsWithoutPriorDecomp(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    T* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    T* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    T* dev_ptr_Hpp_prior,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    T* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    T* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    T* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    T* dev_ptr_H22,
    int num_elem_B11,
    T* dev_ptr_B11,
    int num_elem_B22,
    T* dev_ptr_B22
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_SOLVE_DELTA_CUH
