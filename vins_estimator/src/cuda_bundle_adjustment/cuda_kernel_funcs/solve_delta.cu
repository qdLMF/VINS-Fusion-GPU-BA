//
// Created by lmf on 23-8-1.
//

#include <chrono>
#include <fstream>

#include "solve_delta.cuh"
#include "cublas_funcs.cuh"
#include "cusolver_funcs.cuh"


#define DIAGONAL_MIN 1e-16
#define DIAGONAL_MAX 1e32

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__global__ void HpmHmmInv(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const T* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    const T* dev_ptr_Hpm,
    T* dev_ptr_HpmHmmInv
) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int row_idx = ix;
    unsigned int col_idx = iy;

    if(row_idx < num_rows_Hpm && col_idx < num_cols_Hpm) {
        T a = dev_ptr_Hpm[col_idx * num_rows_Hpm + row_idx];
        T b = dev_ptr_Hmm_diag[col_idx];
        T c = a / b;
        dev_ptr_HpmHmmInv[col_idx * num_rows_Hpm + row_idx] = c;
    }
}
// instantiation
template __global__ void HpmHmmInv<double>(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const double* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    const double* dev_ptr_Hpm,
    double* dev_ptr_HpmHmmInv
);
template __global__ void HpmHmmInv<float>(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const float* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    const float* dev_ptr_Hpm,
    float* dev_ptr_HpmHmmInv
);

// ----------

template<typename T>
__global__ void HpmHmmInvForMarg(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const T* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    T* dev_ptr_Hpm
) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int row_idx = ix;
    unsigned int col_idx = iy;

    if(row_idx < num_rows_Hpm && col_idx < num_cols_Hpm) {
        T a = dev_ptr_Hpm[col_idx * num_rows_Hpm + row_idx];
        T b = dev_ptr_Hmm_diag[col_idx];
        T c = a / b;
        dev_ptr_Hpm[col_idx * num_rows_Hpm + row_idx] = c;
    }
}
// instantiation
template __global__ void HpmHmmInvForMarg<double>(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const double* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    double* dev_ptr_Hpm
);
template __global__ void HpmHmmInvForMarg<float>(  // Hpm is col-major
    int num_elem_Hmm_diag,
    const float* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    float* dev_ptr_Hpm
);

// ----------

template<typename T>
__global__ void AddLambdaV2(  // Hpp is col-major
    const T* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    T* dev_ptr_Hpp,
    T* dev_ptr_Hpp_schur_with_lambda
) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int row_idx = ix;
    unsigned int col_idx = iy;

    if(row_idx >= nrows_Hpp || col_idx >= ncols_Hpp) {
        return;
    }

    T val = dev_ptr_Hpp[col_idx * nrows_Hpp + row_idx];
    if(row_idx == col_idx) {
        val += *dev_ptr_lambda;
    }

    dev_ptr_Hpp_schur_with_lambda[col_idx * nrows_Hpp + row_idx] = val;
}
// instantiation
template __global__ void AddLambdaV2<double>(  // Hpp is col-major
    const double* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    double* dev_ptr_Hpp,
    double* dev_ptr_Hpp_schur_with_lambda
);
template __global__ void AddLambdaV2<float>(  // Hpp is col-major
    const float* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    float* dev_ptr_Hpp,
    float* dev_ptr_Hpp_schur_with_lambda
);

// ----------

template<typename T>
__global__ void AddLambdaToHpp(  // Hpp is col-major
    const T* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    const T* dev_ptr_Hpp,
    T* dev_ptr_Hpp_with_lambda
) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int row_idx = ix;
    unsigned int col_idx = iy;

    if(row_idx >= nrows_Hpp || col_idx >= ncols_Hpp) {
        return;
    }

    T val = dev_ptr_Hpp[col_idx * nrows_Hpp + row_idx];
    if(row_idx == col_idx) {
        val = min(max(val, DIAGONAL_MIN), DIAGONAL_MAX);
        val += (*dev_ptr_lambda) * val;
    }

    dev_ptr_Hpp_with_lambda[col_idx * nrows_Hpp + row_idx] = val;
}
// instantiation
template __global__ void AddLambdaToHpp<double>(  // Hpp is col-major
    const double* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    const double* dev_ptr_Hpp,
    double* dev_ptr_Hpp_with_lambda
);
template __global__ void AddLambdaToHpp<float>(  // Hpp is col-major
    const float* dev_ptr_lambda,
    int nrows_Hpp,
    int ncols_Hpp,
    const float* dev_ptr_Hpp,
    float* dev_ptr_Hpp_with_lambda
);

// ----------

template<typename T>
__global__ void AddLambdaToHmmDiag(  // Hmm_diag
    const T* dev_ptr_lambda,
    int nelem_Hmm_diag,
    const T* dev_ptr_Hmm_diag,
    T* dev_ptr_Hmm_diag_with_lambda
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= nelem_Hmm_diag) {
        return;
    }

    T val = dev_ptr_Hmm_diag[tid];
    val = min(max(val, DIAGONAL_MIN), DIAGONAL_MAX);
    val += (*dev_ptr_lambda) * val;

    dev_ptr_Hmm_diag_with_lambda[tid] = val;
}
// instantiation
template __global__ void AddLambdaToHmmDiag<double>(  // Hmm_diag
    const double* dev_ptr_lambda,
    int nelem_Hmm_diag,
    const double* dev_ptr_Hmm_diag,
    double* dev_ptr_Hmm_diag_with_lambda
);
template __global__ void AddLambdaToHmmDiag<float>(  // Hmm_diag
    const float* dev_ptr_lambda,
    int nelem_Hmm_diag,
    const float* dev_ptr_Hmm_diag,
    float* dev_ptr_Hmm_diag_with_lambda
);

// ----------

template<typename T>
__global__ void SolveDeltaXmm(
    int num_elem_Hmm_diag,
    const T* dev_ptr_Hmm_diag,
    T* dev_ptr_delta_inv_depth
) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < num_elem_Hmm_diag) {
        dev_ptr_delta_inv_depth[idx] = dev_ptr_delta_inv_depth[idx] / dev_ptr_Hmm_diag[idx];
    }
}
// instantiation
template __global__ void SolveDeltaXmm<double>(
    int num_elem_Hmm_diag,
    const double* dev_ptr_Hmm_diag,
    double* dev_ptr_delta_inv_depth
);
template __global__ void SolveDeltaXmm<float>(
    int num_elem_Hmm_diag,
    const float* dev_ptr_Hmm_diag,
    float* dev_ptr_delta_inv_depth
);

// ----------

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
) {
    // ----------------------------------------------------------------------------------------------------
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    cusolverStatus_t cusolver_status;

    int num_threads_x;
    int num_blocks_x;

    int num_threads_y;
    int num_blocks_y;

    dim3 num_blocks;
    dim3 num_threads_per_block;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // add lambda to Hpp
    num_threads_x = 32;
    num_blocks_x = num_cols_Hpp / num_threads_x;
    if( (num_cols_Hpp % num_threads_x) != 0 )
        num_blocks_x += 1;

    num_threads_y = 32;
    num_blocks_y = num_rows_Hpp / num_threads_y;
    if( (num_rows_Hpp % num_threads_y) != 0 )
        num_blocks_y += 1;

    num_blocks = {static_cast<unsigned int>(num_blocks_x), static_cast<unsigned int>(num_blocks_y)};
    num_threads_per_block = {static_cast<unsigned int>(num_threads_x), static_cast<unsigned int>(num_threads_y)};
    AddLambdaToHpp<T><<< num_blocks, num_threads_per_block, 0 >>>(dev_ptr_lambda, num_rows_Hpp, num_cols_Hpp, dev_ptr_Hpp, dev_ptr_Hpp_with_lambda);
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // add lambda to Hmm_diag
    num_threads_x = 32;
    num_blocks_x = num_elem_Hmm_diag / num_threads_x;
    if( (num_elem_Hmm_diag % num_threads_x) != 0 )
        num_blocks_x += 1;

    num_blocks = {static_cast<unsigned int>(num_blocks_x)};
    num_threads_per_block = {static_cast<unsigned int>(num_threads_x)};
    AddLambdaToHmmDiag<T><<< num_blocks, num_threads_per_block, 0 >>>(dev_ptr_lambda, num_elem_Hmm_diag, dev_ptr_Hmm_diag, dev_ptr_Hmm_diag_with_lambda);
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    num_threads_x = 32;
    num_blocks_x = num_rows_Hpm / num_threads_x;
    if( (num_rows_Hpm % num_threads_x) != 0 )
        num_blocks_x += 1;

    num_threads_y = 32;
    num_blocks_y = num_cols_Hpm / num_threads_y;
    if( (num_cols_Hpm % num_threads_y) != 0 )
        num_blocks_y += 1;

    num_blocks = {static_cast<unsigned int>(num_blocks_x), static_cast<unsigned int>(num_blocks_y)};
    num_threads_per_block = {static_cast<unsigned int>(num_threads_x), static_cast<unsigned int>(num_threads_y)};
    HpmHmmInv<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_elem_Hmm_diag,
        dev_ptr_Hmm_diag_with_lambda,
        num_rows_Hpm,
        num_cols_Hpm,
        dev_ptr_Hpm,
        dev_ptr_HpmHmmInv
    );
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int num_rows_HpmHmmInv = num_rows_Hpm;
    int num_cols_HpmHmmInv = num_cols_Hpm;
    int leading_dim_HpmHmmInv = num_rows_HpmHmmInv;

    int leading_dim_Hmp = num_rows_Hmp;
    int leading_dim_Hpp = num_rows_Hpp;
    int leading_dim_Bpp = num_elem_Bpp;
    int leading_dim_Bmm = num_elem_Bmm;

    T alpha = 0.0;
    T beta  = 0.0;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // Hpp <- Hpp - HpmHmmInvHmp
    cudaMemcpy(dev_ptr_Hpp_schur, dev_ptr_Hpp_with_lambda, num_rows_Hpp * num_cols_Hpp * sizeof(T), cudaMemcpyDeviceToDevice);
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_HpmHmmInv,
        num_cols_Hmp,
        num_cols_HpmHmmInv,
        &alpha,
        dev_ptr_HpmHmmInv,
        leading_dim_HpmHmmInv,
        dev_ptr_Hmp,
        leading_dim_Hmp,
        &beta,
        dev_ptr_Hpp_schur,
        leading_dim_Hpp
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // Bpp <- Bpp - HpmHmmInvBmm
    cudaMemcpy(dev_ptr_Bpp_schur, dev_ptr_Bpp, num_elem_Bpp * sizeof(T), cudaMemcpyDeviceToDevice);
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemv_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        num_rows_HpmHmmInv,
        num_cols_HpmHmmInv,
        &alpha,
        dev_ptr_HpmHmmInv,
        leading_dim_HpmHmmInv,
        dev_ptr_Bmm,
        1,
        &beta,
        dev_ptr_Bpp_schur,
        1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // setup workspace for QR decomposition
    int size_workspace_potrf;
    cusolver_status = cusolverDn_potrf_bufferSize<T>(
        cusolver_dense_handle,
        CUBLAS_FILL_MODE_LOWER,
        num_rows_Hpp,
        dev_ptr_Hpp_schur,
        leading_dim_Hpp,
        &size_workspace_potrf
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // QR decomposition, results are stored in Hpp's lower triangle and size_workspace_potrf
    int* dev_info = nullptr;
    cudaMalloc((void**)&dev_info, sizeof(int));
    cudaMemset(dev_info, 0, sizeof(int));
    T* workspace_potrf = nullptr;
    cudaMalloc((void**)&workspace_potrf, sizeof(T) * size_workspace_potrf);
    cudaMemset(workspace_potrf, 0, sizeof(T) * size_workspace_potrf);
    cusolver_status = cusolverDn_potrf<T>(
        cusolver_dense_handle,
        CUBLAS_FILL_MODE_LOWER,
        num_rows_Hpp,
        dev_ptr_Hpp_schur,
        leading_dim_Hpp,
        workspace_potrf,
        size_workspace_potrf,
        dev_info
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // solve Xpp
    cudaMemcpy(dev_ptr_delta_Xpp, dev_ptr_Bpp_schur, num_elem_Bpp * sizeof(T), cudaMemcpyDeviceToDevice);
    cusolver_status = cusolverDn_potrs<T>(
        cusolver_dense_handle,
        CUBLAS_FILL_MODE_LOWER,
        num_rows_Hpp,
        1,
        dev_ptr_Hpp_schur,
        leading_dim_Hpp,
        dev_ptr_delta_Xpp,
        leading_dim_Bpp,
        dev_info
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // solve delta inverse depth, stage 1
    assert(num_elem_Bmm == num_elem_delta_inv_depth);
    cudaMemcpy(dev_ptr_delta_inv_depth, dev_ptr_Bmm, num_elem_Bmm * sizeof(T), cudaMemcpyDeviceToDevice);
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemv_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        num_rows_Hmp,
        num_cols_Hmp,
        &alpha,
        dev_ptr_Hmp,
        leading_dim_Hmp,
        dev_ptr_delta_Xpp,
        1,
        &beta,
        dev_ptr_delta_inv_depth,
        1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // solve delta inverse depth, stage 2
    assert(num_elem_Hmm_diag == num_elem_delta_inv_depth);
    int _num_threads_1d = 32;
    int _num_blocks_1d = num_elem_Hmm_diag / _num_threads_1d;
    if( (num_elem_Hmm_diag % _num_threads_1d) != 0 )
        _num_blocks_1d += 1;
    dim3 num_blocks_1d(_num_blocks_1d);
    dim3 num_threads_per_block_1d(_num_threads_1d);
    SolveDeltaXmm<T><<< num_blocks_1d, num_threads_per_block_1d, 0 >>>(num_elem_Hmm_diag, dev_ptr_Hmm_diag, dev_ptr_delta_inv_depth);
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // free cuslover workspace
    cudaFree(dev_info);
    cudaFree(workspace_potrf);
    // ----------------------------------------------------------------------------------------------------
}
// instantiation
template void SolveDeltaKernels<double>(
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp,
    int num_cols_Hpp,
    double* dev_ptr_Hpp,
    int num_elem_Hmm_diag,
    double* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    double* dev_ptr_Hpm,
    int num_rows_Hmp,
    int num_cols_Hmp,
    double* dev_ptr_Hmp,
    int num_elem_Bpp,
    double* dev_ptr_Bpp,
    int num_elem_Bmm,
    double* dev_ptr_Bmm,
    double* dev_ptr_lambda,
    double* dev_ptr_Hpp_with_lambda,
    double* dev_ptr_Hmm_diag_with_lambda,
    double* dev_ptr_HpmHmmInv,
    double* dev_ptr_Hpp_schur,
    double* dev_ptr_Bpp_schur,
    double* dev_ptr_delta_Xpp,
    double* dev_ptr_delta_inv_depth
);
template void SolveDeltaKernels<float>(
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp,
    int num_cols_Hpp,
    float* dev_ptr_Hpp,
    int num_elem_Hmm_diag,
    float* dev_ptr_Hmm_diag,
    int num_rows_Hpm,
    int num_cols_Hpm,
    float* dev_ptr_Hpm,
    int num_rows_Hmp,
    int num_cols_Hmp,
    float* dev_ptr_Hmp,
    int num_elem_Bpp,
    float* dev_ptr_Bpp,
    int num_elem_Bmm,
    float* dev_ptr_Bmm,
    float* dev_ptr_lambda,
    float* dev_ptr_Hpp_with_lambda,
    float* dev_ptr_Hmm_diag_with_lambda,
    float* dev_ptr_HpmHmmInv,
    float* dev_ptr_Hpp_schur,
    float* dev_ptr_Bpp_schur,
    float* dev_ptr_delta_Xpp,
    float* dev_ptr_delta_inv_depth
);

// ------------------------------------------------------------------------------------------------------------------------

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
) {
    // ----------------------------------------------------------------------------------------------------
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int num_threads_outer_col = 32;
    int num_blocks_outer_col = num_cols_Hpm_for_marg / num_threads_outer_col;
    if( (num_cols_Hpm_for_marg % num_threads_outer_col) != 0 )
        num_blocks_outer_col += 1;

    int num_threads_inner_row = 32;
    int num_blocks_inner_row = num_rows_Hpm_for_marg / num_threads_inner_row;
    if( (num_rows_Hpm_for_marg % num_threads_inner_row) != 0 )
        num_blocks_inner_row += 1;

    dim3 num_blocks(num_blocks_inner_row, num_blocks_outer_col);
    dim3 num_threads_per_block(num_threads_inner_row, num_threads_outer_col);
    HpmHmmInvForMarg<T><<< num_blocks, num_threads_per_block, 0 >>>(
        num_elem_Hmm_diag_for_marg,
        dev_ptr_Hmm_diag_for_marg,
        num_rows_Hpm_for_marg,
        num_cols_Hpm_for_marg,
        dev_ptr_Hpm_for_marg
    );
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    T* dev_ptr_HpmHmmInv_for_marg = dev_ptr_Hpm_for_marg;
    int num_rows_HpmHmmInv_for_marg = num_rows_Hpm_for_marg;
    int num_cols_HpmHmmInv_for_marg = num_cols_Hpm_for_marg;
    int leading_dim_HpmHmmInv_for_marg = num_rows_HpmHmmInv_for_marg;

    int leading_dim_Hmp_for_marg = num_rows_Hmp_for_marg;
    int leading_dim_Hpp_for_marg = num_rows_Hpp_for_marg;
    int leading_dim_Bpp_for_marg = num_elem_Bpp_for_marg;
    int leading_dim_Bmm_for_marg = num_elem_Bmm_for_marg;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    T alpha = -1.0;
    T beta  = 1.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_HpmHmmInv_for_marg,
        num_cols_Hmp_for_marg,
        num_cols_HpmHmmInv_for_marg,
        &alpha,
        dev_ptr_HpmHmmInv_for_marg,
        leading_dim_HpmHmmInv_for_marg,
        dev_ptr_Hmp_for_marg,
        leading_dim_Hmp_for_marg,
        &beta,
        dev_ptr_Hpp_for_marg,
        leading_dim_Hpp_for_marg
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemv_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        num_rows_HpmHmmInv_for_marg,
        num_cols_HpmHmmInv_for_marg,
        &alpha,
        dev_ptr_HpmHmmInv_for_marg,
        leading_dim_HpmHmmInv_for_marg,
        dev_ptr_Bmm_for_marg,
        1,
        &beta,
        dev_ptr_Bpp_for_marg,
        1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------
}
// instantiation
template void MargInvDepthKernels<double>(
    cublasHandle_t& cublas_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    double* dev_ptr_Hpp_for_marg,
    int num_elem_Hmm_diag_for_marg,
    double* dev_ptr_Hmm_diag_for_marg,
    int num_rows_Hpm_for_marg,
    int num_cols_Hpm_for_marg,
    double* dev_ptr_Hpm_for_marg,
    int num_rows_Hmp_for_marg,
    int num_cols_Hmp_for_marg,
    double* dev_ptr_Hmp_for_marg,
    int num_elem_Bpp_for_marg,
    double* dev_ptr_Bpp_for_marg,
    int num_elem_Bmm_for_marg,
    double* dev_ptr_Bmm_for_marg
);
template void MargInvDepthKernels<float>(
    cublasHandle_t& cublas_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    float* dev_ptr_Hpp_for_marg,
    int num_elem_Hmm_diag_for_marg,
    float* dev_ptr_Hmm_diag_for_marg,
    int num_rows_Hpm_for_marg,
    int num_cols_Hpm_for_marg,
    float* dev_ptr_Hpm_for_marg,
    int num_rows_Hmp_for_marg,
    int num_cols_Hmp_for_marg,
    float* dev_ptr_Hmp_for_marg,
    int num_elem_Bpp_for_marg,
    float* dev_ptr_Bpp_for_marg,
    int num_elem_Bmm_for_marg,
    float* dev_ptr_Bmm_for_marg
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
__global__ void Hpp_for_marg_add_prior_and_devide(
    int keyframe_idx,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    T* dev_ptr_Hpp_for_marg,
    int leading_dim_Hpp_for_marg,
    T* dev_ptr_Hpp_prior,
    int leading_dim_Hpp_prior,
    T* dev_ptr_H11,
    int leading_dim_H11,
    T* dev_ptr_H12,
    int leading_dim_H12,
    T* dev_ptr_H21,
    int leading_dim_H21,
    T* dev_ptr_H22,
    int leading_dim_H22
) {
    unsigned int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    if(row_idx >= num_rows_Hpp_for_marg || col_idx >= num_cols_Hpp_for_marg) {
        return;
    }

    T element_1 = __ldg(dev_ptr_Hpp_for_marg + leading_dim_Hpp_for_marg * col_idx + row_idx);
    T element_2 = 0.0;
    if(dev_ptr_Hpp_prior) {
        element_2 = __ldg(dev_ptr_Hpp_prior + leading_dim_Hpp_prior * col_idx + row_idx);
    }

    int area_border = 13 + keyframe_idx * 15;

    unsigned char area_idx_row = row_idx < area_border ? 0 : (row_idx < (area_border + 15) ? 1 : 2);
    unsigned char area_idx_col = col_idx < area_border ? 0 : (col_idx < (area_border + 15) ? 1 : 2);

    unsigned int dst_row_idx = area_idx_row == 0 ? row_idx : ((area_idx_row == 1) ? (row_idx - area_border) : (row_idx - 15));
    unsigned int dst_col_idx = area_idx_col == 0 ? col_idx : ((area_idx_col == 1) ? (col_idx - area_border) : (col_idx - 15));

    T* dst_ptr;
    int leading_dim;
    if(    (area_idx_row == 0 && area_idx_col == 0) || (area_idx_row == 0 && area_idx_col == 2)
           || (area_idx_row == 2 && area_idx_col == 0) || (area_idx_row == 2 && area_idx_col == 2) ) {
        dst_ptr = dev_ptr_H11;
        leading_dim = leading_dim_H11;
    } else if( (area_idx_row == 0 && area_idx_col == 1) || (area_idx_row == 2 && area_idx_col == 1) ) {
        dst_ptr = dev_ptr_H12;
        leading_dim = leading_dim_H12;
    } else if( (area_idx_row == 1 && area_idx_col == 0) || (area_idx_row == 1 && area_idx_col == 2) ) {
        dst_ptr = dev_ptr_H21;
        leading_dim = leading_dim_H21;
    } else {
        dst_ptr = dev_ptr_H22;
        leading_dim = leading_dim_H22;
    }

    *(dst_ptr + leading_dim * dst_col_idx + dst_row_idx) = element_1 + element_2;

    *(dev_ptr_Hpp_for_marg + leading_dim_Hpp_for_marg * col_idx + row_idx) = element_1 + element_2;
}
// instantiation
template __global__ void Hpp_for_marg_add_prior_and_devide<double>(
    int keyframe_idx,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    double* dev_ptr_Hpp_for_marg,
    int leading_dim_Hpp_for_marg,
    double* dev_ptr_Hpp_prior,
    int leading_dim_Hpp_prior,
    double* dev_ptr_H11,
    int leading_dim_H11,
    double* dev_ptr_H12,
    int leading_dim_H12,
    double* dev_ptr_H21,
    int leading_dim_H21,
    double* dev_ptr_H22,
    int leading_dim_H22
);
template __global__ void Hpp_for_marg_add_prior_and_devide<float>(
    int keyframe_idx,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    float* dev_ptr_Hpp_for_marg,
    int leading_dim_Hpp_for_marg,
    float* dev_ptr_Hpp_prior,
    int leading_dim_Hpp_prior,
    float* dev_ptr_H11,
    int leading_dim_H11,
    float* dev_ptr_H12,
    int leading_dim_H12,
    float* dev_ptr_H21,
    int leading_dim_H21,
    float* dev_ptr_H22,
    int leading_dim_H22
);

// ------------------------------------------------------------------------------------------------------------------------

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
) {
    // ----------------------------------------------------------------------------------------------------
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    cusolverStatus_t cusolver_status;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int num_threads_x = 32;
    int num_blocks_x = num_cols_Hpp_for_marg / num_threads_x;
    if( (num_cols_Hpp_for_marg % num_threads_x) != 0 )
        num_blocks_x += 1;

    int num_threads_y = 8;
    int num_blocks_y = num_rows_Hpp_for_marg / num_threads_y;
    if( (num_rows_Hpp_for_marg % num_threads_y) != 0 )
        num_blocks_y += 1;

    dim3 num_blocks(num_blocks_x, num_blocks_y);
    dim3 num_threads_per_block(num_threads_x, num_threads_y);
    Hpp_for_marg_add_prior_and_devide<T><<< num_blocks, num_threads_per_block, 0 >>>(
        keyframe_idx,
        num_rows_Hpp_for_marg,
        num_cols_Hpp_for_marg,
        dev_ptr_Hpp_for_marg,
        num_rows_Hpp_for_marg,
        dev_ptr_Hpp_prior,
        num_rows_Hpp_for_marg,
        dev_ptr_H11,
        num_rows_H11,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H21,
        num_rows_H21,
        dev_ptr_H22,
        num_rows_H22
    );
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_H22;
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_H22_inv;
    host_H22.resize(num_rows_H22, num_cols_H22);
    cudaMemcpy(host_H22.data(), dev_ptr_H22, num_rows_H22 * num_cols_H22 * sizeof(T), cudaMemcpyDeviceToHost);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> saes(host_H22);
    host_H22_inv = \
    saes.eigenvectors() \
    * Eigen::Matrix<T, Eigen::Dynamic, 1>( (saes.eigenvalues().array() > 1e-8).select(saes.eigenvalues().array().inverse(), 0) ).asDiagonal()   \
    * saes.eigenvectors().transpose();
    cudaMemcpy(dev_ptr_H22, host_H22_inv.data(), num_rows_H22 * num_cols_H22 * sizeof(T), cudaMemcpyHostToDevice);

    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bpp_for_marg;
    host_Bpp_for_marg.resize(num_elem_Bpp_for_marg, 1);
    cudaMemcpy(host_Bpp_for_marg.data(), dev_ptr_Bpp_for_marg, num_elem_Bpp_for_marg * sizeof(T), cudaMemcpyDeviceToHost);
    if(host_Bprior.rows() == host_Bpp_for_marg.rows() && host_Bprior.cols() == host_Bpp_for_marg.cols()) {
        host_Bpp_for_marg += host_Bprior;
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> e = host_Bpp_for_marg.block(13 + 15 * keyframe_idx, 0, 15, 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f = host_Bpp_for_marg.block(13 + 15 * keyframe_idx + 15, 0, host_Bpp_for_marg.rows() - (13 + 15 * keyframe_idx + 15), 1);
    host_Bpp_for_marg.block(13 + 15 * keyframe_idx, 0, host_Bpp_for_marg.rows() - (13 + 15 * keyframe_idx + 15), 1) = f;
    host_Bpp_for_marg.block(host_Bpp_for_marg.rows() - 15, 0, 15, 1) = e;
    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_B11;
    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_B22;
    host_B11 = host_Bpp_for_marg.block(0, 0, num_elem_B11, 1);
    host_B22 = host_Bpp_for_marg.block(num_elem_B11, 0, num_elem_B22, 1);
    cudaMemcpy(dev_ptr_Bpp_for_marg, host_Bpp_for_marg.data(), num_elem_Bpp_for_marg * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_B11, host_B11.data(), num_elem_B11 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_B22, host_B22.data(), num_elem_B22 * sizeof(T), cudaMemcpyHostToDevice);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (H12 * H22) -> H12
    T alpha = 1.0;
    T beta  = 0.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H22,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H22,    // is H22.inverse()
        num_rows_H22,
        &beta,
        dev_ptr_H12,
        num_rows_H12
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (H11 - H12 * H22 * H21) -> H11
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H21,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H21,
        num_rows_H21,
        &beta,
        dev_ptr_H11,
        num_rows_H11
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (B11 - H12 * H22 * B21) -> B11
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemv_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_B22,
        1,
        &beta,
        dev_ptr_B11,
        1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    cudaMemcpy(dev_ptr_Hprior_eigenvec, dev_ptr_H11, sizeof(T) * num_rows_H11 * num_cols_H11, cudaMemcpyDeviceToDevice);
    cudaMemset(dev_ptr_Hprior_eigenval, 0, sizeof(T) * num_rows_H11);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int size_workspace_syevd;
    cusolver_status = cusolverDn_syevd_bufferSize<T>(
        cusolver_dense_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        num_rows_H11,
        dev_ptr_Hprior_eigenvec,
        num_rows_H11,
        dev_ptr_Hprior_eigenval,
        &size_workspace_syevd
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int* dev_info = nullptr;
    cudaMalloc((void**)&dev_info, sizeof(int));
    cudaMemset(dev_info, 0, sizeof(int));
    T* workspace_syevd = nullptr;
    cudaMalloc((void**)&workspace_syevd, sizeof(T) * size_workspace_syevd);
    cudaMemset(workspace_syevd, 0, sizeof(T) * size_workspace_syevd);
    cusolver_status = cusolverDn_syevd<T>(
        cusolver_dense_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        num_rows_H11,
        dev_ptr_Hprior_eigenvec,
        num_rows_H11,
        dev_ptr_Hprior_eigenval,
        workspace_syevd,
        size_workspace_syevd,
        dev_info
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    cudaFree(dev_info);
    cudaFree(workspace_syevd);
    // ----------------------------------------------------------------------------------------------------
}
// instantiation
template void MargKeyFrameKernelsWithPriorDecomp<double>(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    double* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    double* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    double* dev_ptr_Hpp_prior,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    double* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    double* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    double* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    double* dev_ptr_H22,
    int num_elem_B11,
    double* dev_ptr_B11,
    int num_elem_B22,
    double* dev_ptr_B22,
    double* dev_ptr_Hprior_eigenvec,
    double* dev_ptr_Hprior_eigenval
);
template void MargKeyFrameKernelsWithPriorDecomp<float>(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    float* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    float* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    float* dev_ptr_Hpp_prior,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    float* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    float* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    float* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    float* dev_ptr_H22,
    int num_elem_B11,
    float* dev_ptr_B11,
    int num_elem_B22,
    float* dev_ptr_B22,
    float* dev_ptr_Hprior_eigenvec,
    float* dev_ptr_Hprior_eigenval
);

// ------------------------------------------------------------------------------------------------------------------------

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
) {
    // ----------------------------------------------------------------------------------------------------
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    cusolverStatus_t cusolver_status;
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    int num_threads_x = 32;
    int num_blocks_x = num_cols_Hpp_for_marg / num_threads_x;
    if( (num_cols_Hpp_for_marg % num_threads_x) != 0 )
        num_blocks_x += 1;

    int num_threads_y = 8;
    int num_blocks_y = num_rows_Hpp_for_marg / num_threads_y;
    if( (num_rows_Hpp_for_marg % num_threads_y) != 0 )
        num_blocks_y += 1;

    dim3 num_blocks(num_blocks_x, num_blocks_y);
    dim3 num_threads_per_block(num_threads_x, num_threads_y);
    Hpp_for_marg_add_prior_and_devide<T><<< num_blocks, num_threads_per_block, 0 >>>(
        keyframe_idx,
        num_rows_Hpp_for_marg,
        num_cols_Hpp_for_marg,
        dev_ptr_Hpp_for_marg,
        num_rows_Hpp_for_marg,
        dev_ptr_Hpp_prior,
        num_rows_Hpp_for_marg,
        dev_ptr_H11,
        num_rows_H11,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H21,
        num_rows_H21,
        dev_ptr_H22,
        num_rows_H22
    );
    cuda_status = cudaStreamSynchronize(0);
    assert(cudaSuccess == cuda_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_H22;
    static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_H22_inv;
    host_H22.resize(num_rows_H22, num_cols_H22);
    cudaMemcpy(host_H22.data(), dev_ptr_H22, num_rows_H22 * num_cols_H22 * sizeof(T), cudaMemcpyDeviceToHost);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> saes(host_H22);
    host_H22_inv = \
    saes.eigenvectors()                                                                                                     \
    * Eigen::Matrix<T, Eigen::Dynamic, 1>( (saes.eigenvalues().array() > 1e-8).select(saes.eigenvalues().array().inverse(), 0) ).asDiagonal()   \
    * saes.eigenvectors().transpose();
    cudaMemcpy(dev_ptr_H22, host_H22_inv.data(), num_rows_H22 * num_cols_H22 * sizeof(T), cudaMemcpyHostToDevice);

    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bpp_for_marg;
    host_Bpp_for_marg.resize(num_elem_Bpp_for_marg, 1);
    cudaMemcpy(host_Bpp_for_marg.data(), dev_ptr_Bpp_for_marg, num_elem_Bpp_for_marg * sizeof(T), cudaMemcpyDeviceToHost);
    if(host_Bprior.rows() == host_Bpp_for_marg.rows() && host_Bprior.cols() == host_Bpp_for_marg.cols()) {
        host_Bpp_for_marg += host_Bprior;
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> e = host_Bpp_for_marg.block(13 + 15 * keyframe_idx, 0, 15, 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f = host_Bpp_for_marg.block(13 + 15 * keyframe_idx + 15, 0, host_Bpp_for_marg.rows() - (13 + 15 * keyframe_idx + 15), 1);
    host_Bpp_for_marg.block(13 + 15 * keyframe_idx, 0, host_Bpp_for_marg.rows() - (13 + 15 * keyframe_idx + 15), 1) = f;
    host_Bpp_for_marg.block(host_Bpp_for_marg.rows() - 15, 0, 15, 1) = e;
    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_B11;
    static Eigen::Matrix<T, Eigen::Dynamic, 1> host_B22;
    host_B11 = host_Bpp_for_marg.block(0, 0, num_elem_B11, 1);
    host_B22 = host_Bpp_for_marg.block(num_elem_B11, 0, num_elem_B22, 1);
    cudaMemcpy(dev_ptr_Bpp_for_marg, host_Bpp_for_marg.data(), num_elem_Bpp_for_marg * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_B11, host_B11.data(), num_elem_B11 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr_B22, host_B22.data(), num_elem_B22 * sizeof(T), cudaMemcpyHostToDevice);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (H12 * H22) -> H12
    T alpha = 1.0;
    T beta  = 0.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H22,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H22,    // is H22.inverse()
        num_rows_H22,
        &beta,
        dev_ptr_H12,
        num_rows_H12
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (H11 - H12 * H22 * H21) -> H11
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemm_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H21,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_H21,
        num_rows_H21,
        &beta,
        dev_ptr_H11,
        num_rows_H11
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // (B11 - H12 * H22 * B21) -> B11
    alpha = -1.0;
    beta  = 1.0;
    cublas_status = cublas_gemv_v2<T>(
        cublas_handle,
        CUBLAS_OP_N,
        num_rows_H12,
        num_cols_H12,
        &alpha,
        dev_ptr_H12,
        num_rows_H12,
        dev_ptr_B22,
        1,
        &beta,
        dev_ptr_B11,
        1
    );
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // ----------------------------------------------------------------------------------------------------
}
// instantiation
template void MargKeyFrameKernelsWithoutPriorDecomp<double>(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    double* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    double* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    double* dev_ptr_Hpp_prior,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    double* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    double* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    double* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    double* dev_ptr_H22,
    int num_elem_B11,
    double* dev_ptr_B11,
    int num_elem_B22,
    double* dev_ptr_B22
);
template void MargKeyFrameKernelsWithoutPriorDecomp<float>(
    int keyframe_idx,
    cublasHandle_t& cublas_handle,
    cusolverDnHandle_t& cusolver_dense_handle,
    int num_rows_Hpp_for_marg,
    int num_cols_Hpp_for_marg,
    float* dev_ptr_Hpp_for_marg,
    int num_elem_Bpp_for_marg,
    float* dev_ptr_Bpp_for_marg,
    int num_rows_Hpp_prior,
    int num_cols_Hpp_prior,
    float* dev_ptr_Hpp_prior,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& host_Bprior,
    int num_rows_H11,
    int num_cols_H11,
    float* dev_ptr_H11,
    int num_rows_H12,
    int num_cols_H12,
    float* dev_ptr_H12,
    int num_rows_H21,
    int num_cols_H21,
    float* dev_ptr_H21,
    int num_rows_H22,
    int num_cols_H22,
    float* dev_ptr_H22,
    int num_elem_B11,
    float* dev_ptr_B11,
    int num_elem_B22,
    float* dev_ptr_B22
);

} // namespace VINS_FUSION_CUDA_BA

