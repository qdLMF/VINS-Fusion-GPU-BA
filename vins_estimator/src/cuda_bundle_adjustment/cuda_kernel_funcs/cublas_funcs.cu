//
// Created by lmf on 23-10-23.
//

#include "cublas_funcs.cuh"

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
) { cublasStatus_t cublas_status_local; return cublas_status_local; }
// specialization
template<>
cublasStatus_t cublas_gemv_v2<double>(
    const cublasHandle_t& handle,
    cublasOperation_t trans,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double* beta,
    double* y, int incy
) {
    return cublasDgemv_v2(
        handle,
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy
    );
}
template<>
cublasStatus_t cublas_gemv_v2<float>(
    const cublasHandle_t& handle,
    cublasOperation_t trans,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy
) {
    return cublasSgemv_v2(
        handle,
        trans,
        m,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy
    );
}

// ----------

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
) { cublasStatus_t cublas_status_local; return cublas_status_local; }
// specialization
template<>
cublasStatus_t cublas_gemm_v2<double>(
    const cublasHandle_t& handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    return cublasDgemm_v2(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc
    );
}
template<>
cublasStatus_t cublas_gemm_v2<float>(
    const cublasHandle_t& handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
) {
    return cublasSgemm_v2(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc
    );
}

} // namespace VINS_FUSION_CUDA_BA
