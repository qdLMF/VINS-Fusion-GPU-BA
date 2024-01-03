//
// Created by lmf on 23-7-22.
//

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "device_utils.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
__device__ Eigen::Matrix<T, 3, 3, Eigen::RowMajor> UtilitySkewSymmetric(const Eigen::Matrix<T, 3, 1>& q) {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> return_value;
    return_value(0, 0) =   0.0; return_value(0, 1) = -q(2); return_value(0, 2) =  q(1);
    return_value(1, 0) =  q(2); return_value(1, 1) =   0.0; return_value(1, 2) = -q(0);
    return_value(2, 0) = -q(1); return_value(2, 1) =  q(0); return_value(2, 2) =   0.0;
    return return_value;
}
// instantiation
template __device__ Eigen::Matrix<double, 3, 3, Eigen::RowMajor> UtilitySkewSymmetric<double>(const Eigen::Matrix<double, 3, 1>& q);
template __device__ Eigen::Matrix<float, 3, 3, Eigen::RowMajor> UtilitySkewSymmetric<float>(const Eigen::Matrix<float, 3, 1>& q);

// ----------

template<typename T>
__device__ Eigen::Quaternion<T> UtilityDeltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
    Eigen::Quaternion<T> return_value;
    return_value.w() = 1.0;
    return_value.x() = theta.x() / 2.0;
    return_value.y() = theta.y() / 2.0;
    return_value.z() = theta.z() / 2.0;
    return return_value;
}
// instantiation
template __device__ Eigen::Quaternion<double> UtilityDeltaQ<double>(const Eigen::Matrix<double, 3, 1>& theta);
template __device__ Eigen::Quaternion<float> UtilityDeltaQ<float>(const Eigen::Matrix<float, 3, 1>& theta);

// ----------

template<typename T>
__device__ Eigen::Matrix<T, 4, 4, Eigen::RowMajor> UtilityQLeft(const Eigen::Quaternion<T>& q) {
    Eigen::Matrix<T, 4, 4, Eigen::RowMajor> return_value;
    return_value(0, 0) = q.w();
    return_value.template block<1, 3>(0, 1) = -q.vec();
    return_value.template block<3, 1>(1, 0) =  q.vec();
    return_value.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity() + UtilitySkewSymmetric<T>(q.vec());
    return return_value;
}
// instantiation
template __device__ Eigen::Matrix<double, 4, 4, Eigen::RowMajor> UtilityQLeft<double>(const Eigen::Quaternion<double>& q);
template __device__ Eigen::Matrix<float, 4, 4, Eigen::RowMajor> UtilityQLeft<float>(const Eigen::Quaternion<float>& q);

// ----------

template<typename T>
__device__ Eigen::Matrix<T, 4, 4, Eigen::RowMajor> UtilityQRight(const Eigen::Quaternion<T>& p) {
    Eigen::Matrix<T, 4, 4, Eigen::RowMajor> return_value;
    return_value(0, 0) = p.w();
    return_value.template block<1, 3>(0, 1) = -p.vec();
    return_value.template block<3, 1>(1, 0) =  p.vec();
    return_value.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<T, 3, 3, Eigen::RowMajor>::Identity() - UtilitySkewSymmetric<T>(p.vec());
    return return_value;
}
// instantiation
template __device__ Eigen::Matrix<double, 4, 4, Eigen::RowMajor> UtilityQRight<double>(const Eigen::Quaternion<double>& p);
template __device__ Eigen::Matrix<float, 4, 4, Eigen::RowMajor> UtilityQRight<float>(const Eigen::Quaternion<float>& p);

// ----------

template<typename T>
__device__ void cauchy_loss(T delta, T err2, Eigen::Matrix<T, 3, 1>& rho) {
    T dsqr = delta * delta;         // c^2
    T dsqrReci = 1. / dsqr;         // 1/c^2
    T aux = dsqrReci * err2 + 1.0;  // 1 + e^2/c^2

    rho[0] = dsqr * log(aux);   // c^2 * log( 1 + e^2/c^2 )
    rho[1] = 1.0 / aux;
    rho[2] = -dsqrReci * pow(rho[1], 2);
}
// instantiation
template __device__ void cauchy_loss<double>(double delta, double err2, Eigen::Matrix<double, 3, 1>& rho);
template __device__ void cauchy_loss<float>(float delta, float err2, Eigen::Matrix<float, 3, 1>& rho);

// ----------

template<typename T>
__device__ void huber_loss(T delta, T err2, Eigen::Matrix<T, 3, 1>& rho) {
    T dsqr = delta * delta; // c^2

    if( err2 <= dsqr ) {    // inlier
        rho[0] = err2;
        rho[1] = 1.0;
        rho[2] = 0.0;
    } else {                // outlier
        T sqrte = sqrt(err2);               // absolute value of the error
        rho[0] = 2 * sqrte * delta - dsqr;  // rho(e)   = 2 * delta * e^(1/2) - delta^2
        rho[1] = delta / sqrte;             // rho'(e)  = delta / sqrt(e)
        rho[2] = -0.5 * rho[1] / err2;      // rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
    }
}
// instantiation
template __device__ void huber_loss<double>(double delta, double err2, Eigen::Matrix<double, 3, 1>& rho);
template __device__ void huber_loss<float>(float delta, float err2, Eigen::Matrix<float, 3, 1>& rho);

// ----------

__device__ BlockRange GetJTJBlockRange(const BlockRange& left, const BlockRange& right) {
    return BlockRange{
        left.col_start,
        left.col_end,
        right.col_start,
        right.col_end
    };
}

// ----------

template<typename T>
__device__ T MyAtomicAdd(T* address, T val) { return 0.0; }

// specialization
template<>
__device__ double MyAtomicAdd<double>(double* address, double val) {
#if __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, 
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    return atomicAdd(address, val);
#endif
}
template<>
__device__ float MyAtomicAdd<float>(float* address, float val) {
    return atomicAdd(address, val);
}

} // namespace VINS_FUSION_CUDA_BA
