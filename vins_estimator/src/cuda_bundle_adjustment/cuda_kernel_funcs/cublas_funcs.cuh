//
// Created by lmf on 23-10-23.
//

#ifndef CUDA_BA_CUBLAS_FUNCS_CUH
#define CUDA_BA_CUBLAS_FUNCS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace VINS_FUSION_CUDA_BA {

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cublasStatus_t>
cublas_gemv_v2(
    const cublasHandle_t& handle,
    cublasOperation_t trans,
    int m, int n,
    const T* alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T* beta,
    T* y, int incy
);

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cublasStatus_t>
cublas_gemm_v2(
    const cublasHandle_t& handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const T* alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T* beta,
    T* C, int ldc
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_CUBLAS_FUNCS_CUH
