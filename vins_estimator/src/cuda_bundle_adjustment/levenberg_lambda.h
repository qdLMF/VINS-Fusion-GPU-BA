//
// Created by lmf on 23-7-31.
//

#ifndef CUDA_BA_LEVENBERG_LAMBDA_H
#define CUDA_BA_LEVENBERG_LAMBDA_H

#include <cuda_runtime.h>

#include <eigen3/Eigen/Core>

#include "device_memory/imu_dev_ptr_set.h"
#include "device_memory/proj_2f1c_dev_ptr_set.h"
#include "device_memory/proj_2f2c_dev_ptr_set.h"
#include "device_memory/proj_1f2c_dev_ptr_set.h"

#include "shape_manager.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
struct LevenbergLambda {
public :
    int num_imu_factors;
    int num_key_frames;
    int num_proj_2f1c_factors;
    int num_proj_2f2c_factors;
    int num_proj_1f2c_factors;
    int num_world_points;

public :
    const IMUFactorDevPtrSet<T>* imu_dev_ptr;
    const Proj2F1CFactorDevPtrSet<T>* proj_2f1c_dev_ptr;
    const Proj2F2CFactorDevPtrSet<T>* proj_2f2c_dev_ptr;
    const Proj1F2CFactorDevPtrSet<T>* proj_1f2c_dev_ptr;

public :
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_imu_chi2;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_proj_2f1c_chi2;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_proj_2f2c_chi2;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_proj_1f2c_chi2;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_err_prior;

public :
    T* dev_ptr_lambda;
    T* dev_ptr_Hpp;
    T* dev_ptr_Hmm_diag;
    T* dev_ptr_Hpp_with_lambda;
    T* dev_ptr_Hmm_diag_with_lambda;

public :
    T* dev_ptr_delta_ex_para_0 = nullptr;      // 3 + 3
    T* dev_ptr_delta_ex_para_1 = nullptr;      // 3 + 3
    T* dev_ptr_delta_states    = nullptr;      // num_key_frames * 15
    T* dev_ptr_delta_inv_depth = nullptr;      // num_world_points
    T* dev_ptr_delta_cur_td    = nullptr;      // 1
    T* dev_ptr_Bpp = nullptr;
    T* dev_ptr_Bmm = nullptr;
    T* dev_ptr_Bprior= nullptr;

public :
    T* dev_ptr_err_prior = nullptr;

public :
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_delta_ex_para_0;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_delta_ex_para_1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_delta_states;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_delta_inv_depth;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_delta_cur_td;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hpp;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_Hmm_diag;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bpp;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bmm;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bprior;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public :
    T currentLambda = 0.0;
    T stopThresholdLM = 0.0;
    T currentChi = 0.0;
    T maxDiagonal = 0;
    T tau = 1e-16;
    int ni = 2;
    T eta = 1e-8;
    int num_consecutive_bad_steps = 0;

public :
    ShapeManager* shapes;

public :
    void ComputeLambdaInitLMV2();
    bool IsGoodStepInLM();
    void Reset();
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_LEVENBERG_LAMBDA_H
