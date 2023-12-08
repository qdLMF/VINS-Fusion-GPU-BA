//
// Created by lmf on 23-10-23.
//

#include "cusolver_funcs.cuh"

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
) { cusolverStatus_t cusolver_status_local; return cusolver_status_local; }
// specialization
template<>
cusolverStatus_t cusolverDn_potrf_bufferSize<double>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    int* Lwork
) {
    return cusolverDnDpotrf_bufferSize(
        handle,
        uplo,
        n,
        A,
        lda,
        Lwork
    );
}
template<>
cusolverStatus_t cusolverDn_potrf_bufferSize<float>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    int* Lwork
) {
    return cusolverDnSpotrf_bufferSize(
        handle,
        uplo,
        n,
        A,
        lda,
        Lwork
    );
}

// ----------

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
) { cusolverStatus_t cusolver_status_local; return cusolver_status_local; }
// specialization
template<>
cusolverStatus_t cusolverDn_potrf<double>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* Workspace,
    int Lwork,
    int* devInfo
) {
    return cusolverDnDpotrf(
        handle,
        uplo,
        n,
        A,
        lda,
        Workspace,
        Lwork,
        devInfo
    );
}
template<>
cusolverStatus_t cusolverDn_potrf<float>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* Workspace,
    int Lwork,
    int* devInfo
) {
    return cusolverDnSpotrf(
        handle,
        uplo,
        n,
        A,
        lda,
        Workspace,
        Lwork,
        devInfo
    );
}

// ----------

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
) { cusolverStatus_t cusolver_status_local; return cusolver_status_local; }
// specialization
template<>
cusolverStatus_t cusolverDn_potrs<double>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    double* A,
    int lda,
    double* B,
    int ldb,
    int* devInfo
) {
    return cusolverDnDpotrs(
        handle,
        uplo,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        devInfo
    );
}
template<>
cusolverStatus_t cusolverDn_potrs<float>(
    const cusolverDnHandle_t& handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    float* A,
    int lda,
    float* B,
    int ldb,
    int* devInfo
) {
    return cusolverDnSpotrs(
        handle,
        uplo,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        devInfo
    );
}

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
) { cusolverStatus_t cusolver_status_local; return cusolver_status_local; }
// specialization
template<>
cusolverStatus_t cusolverDn_syevd_bufferSize<double>(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    int* lwork
) {
    return cusolverDnDsyevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        A,
        lda,
        W,
        lwork
    );
}
template<>
cusolverStatus_t cusolverDn_syevd_bufferSize<float>(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    int* lwork
) {
    return cusolverDnSsyevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        A,
        lda,
        W,
        lwork
    );
}

// ----------

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
) { cusolverStatus_t cusolver_status_local; return cusolver_status_local; }
// specialization
template<>
cusolverStatus_t cusolverDn_syevd<double>(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* devInfo
) {
    return cusolverDnDsyevd(
        handle,
        jobz,
        uplo,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        devInfo
    );
}
template<>
cusolverStatus_t cusolverDn_syevd<float>(
    const cusolverDnHandle_t& handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* devInfo
) {
    return cusolverDnSsyevd(
        handle,
        jobz,
        uplo,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        devInfo
    );
}

} // namespace VINS_FUSION_CUDA_BA





