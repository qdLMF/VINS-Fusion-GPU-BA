//
// Created by lmf on 23-10-23.
//

#ifndef CUDA_BA_CUSOLVER_FUNCS_CUH
#define CUDA_BA_CUSOLVER_FUNCS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace VINS_FUSION_CUDA_BA {

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cusolverStatus_t>
cusolverDn_potrf_bufferSize(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    T* A,
    int lda,
    int* Lwork
);

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cusolverStatus_t>
cusolverDn_potrf(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    T* A,
    int lda,
    T* Workspace,
    int Lwork,
    int* devInfo
);

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cusolverStatus_t>
cusolverDn_potrs(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    T* A,
    int lda,
    T* B,
    int ldb,
    int* devInfo
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cusolverStatus_t>
cusolverDn_syevd_bufferSize(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    T* A,
    int lda,
    T* W,
    int* lwork
);

template<typename T>
std::enable_if_t<(std::is_same<T, double>::value || std::is_same<T, float>::value), cusolverStatus_t>
cusolverDn_syevd(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    T* A,
    int lda,
    T* W,
    T* work,
    int lwork,
    int* devInfo
);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_CUSOLVER_FUNCS_CUH
