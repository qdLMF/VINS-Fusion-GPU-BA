//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_SLIDING_WINDOW_H
#define CUDA_BA_SLIDING_WINDOW_H

#include <map>
#include <set>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "factors.h"
#include "imu_allocator.h"
#include "proj_allocators.h"
#include "shape_manager.h"
#include "levenberg_lambda.h"
#include "utility.h"

#include "device_memory/imu_gpu_mem_pool.h"
#include "device_memory/proj_2f1c_gpu_mem_pool.h"
#include "device_memory/proj_2f2c_gpu_mem_pool.h"
#include "device_memory/proj_1f2c_gpu_mem_pool.h"

#include "cuda_kernel_funcs/update_states.cuh"
#include "cuda_kernel_funcs/launch_kernels.cuh"
#include "cuda_kernel_funcs/solve_delta.cuh"

#include "cuda_streams/block_range_cuda_stream_set.cuh"
#include "cuda_streams/proj_temp_cuda_stream_set.cuh"
#include "cuda_streams/jacobian_residual_cuda_stream_set.cuh"
#include "cuda_streams/hessian_rhs_cuda_stream_set.cuh"
#include "cuda_streams/robust_info_cuda_stream_set.cuh"
#include "cuda_streams/update_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
class SlidingWindow {
private :
    int num_key_frames;
    int max_num_imu_factors;
    int max_num_world_points = 1024;

public :
    bool decompose_prior;

public :
    int marg_keyframe_idx = 0;

public :
    std::vector< SimpleIMUFactor<T> > imu_factor_vec;
    std::vector< SimpleProj2F1CFactor<T> > proj_2f1c_factor_vec;
    std::vector< SimpleProj2F2CFactor<T> > proj_2f2c_factor_vec;
    std::vector< SimpleProj1F2CFactor<T> > proj_1f2c_factor_vec;

public :
    IMUFactorAllocator<T> imu_allocator;
    Proj2F1CFactorAllocator<T> proj_2f1c_allocator;
    Proj2F2CFactorAllocator<T> proj_2f2c_allocator;
    Proj1F2CFactorAllocator<T> proj_1f2c_allocator;

public :
    IMUGPUMemPool<T, 32> imu_gpu_mem_pool;
    Proj2F1CGPUMemPool<T, 8192> proj_2f1c_gpu_mem_pool;
    Proj2F2CGPUMemPool<T, 8192> proj_2f2c_gpu_mem_pool;
    Proj1F2CGPUMemPool<T, 1024> proj_1f2c_gpu_mem_pool;

public :
    ShapeManager shapes;

public :    // states in device
    T* dev_ptr_ex_para_0;   // 3 + 4
    T* dev_ptr_ex_para_1;   // 3 + 4
    T* dev_ptr_states;      // num_key_frames * 16
    T* dev_ptr_inv_depth;   // num_world_points
    T* dev_ptr_cur_td;

public :    // deltas in device
    T* dev_ptr_delta_ex_para_0;     // 3 + 3
    T* dev_ptr_delta_ex_para_1;     // 3 + 3
    T* dev_ptr_delta_states;        // num_key_frames * 15
    T* dev_ptr_delta_inv_depth;     // num_world_points
    T* dev_ptr_delta_cur_td;

public :    // backup states in device
    T* dev_ptr_ex_para_0_backup;    // 3 + 4
    T* dev_ptr_ex_para_1_backup;    // 3 + 4
    T* dev_ptr_states_backup;       // num_key_frames * 16
    T* dev_ptr_inv_depth_backup;    // num_world_points
    T* dev_ptr_cur_td_backup;

public :    // init states in device
    T* dev_ptr_ex_para_0_init;  // 3 + 4
    T* dev_ptr_ex_para_1_init;  // 3 + 4
    T* dev_ptr_states_init;     // num_key_frames * 16
    T* dev_ptr_inv_depth_init;  // num_world_points
    T* dev_ptr_cur_td_init;

public :    // states in host
    Eigen::Matrix<T, 7, 1> host_ex_para_0;              // 3 + 4
    Eigen::Matrix<T, 7, 1> host_ex_para_1;              // 3 + 4
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_states;    // num_key_frames * 16
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_inv_depth; // num_world_points
    Eigen::Matrix<T, 1, 1> host_cur_td;

public :    // state backups in host
    Eigen::Matrix<T, 7, 1> host_ex_para_0_backup;               // 3 + 4
    Eigen::Matrix<T, 7, 1> host_ex_para_1_backup;               // 3 + 4
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_states_backup;     // 3 + 4
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_inv_depth_backup;  // num_world_points
    Eigen::Matrix<T, 1, 1> host_cur_td_backup;

public :    // lambda in device
    T* dev_ptr_lambda;

public :    // hessian and rhs in device
    T* dev_ptr_Hpp;
    T* dev_ptr_Hmm_diag;
    T* dev_ptr_Hpm;
    T* dev_ptr_Hmp;

    T* dev_ptr_Bpp;
    T* dev_ptr_Bmm;

public :    // intermediate results
    T* dev_ptr_Hpp_with_lambda;
    T* dev_ptr_Hmm_diag_with_lambda;
    T* dev_ptr_HpmHmmInv;
    T* dev_ptr_Hpp_schur;
    T* dev_ptr_Bpp_schur;
    T* dev_ptr_delta_Xpp;

public :    // hessian and rhs for marg in device
    T* dev_ptr_Hpp_for_marg;
    T* dev_ptr_Hmm_diag_for_marg;
    T* dev_ptr_Hpm_for_marg;
    T* dev_ptr_Hmp_for_marg;

    T* dev_ptr_Bpp_for_marg;
    T* dev_ptr_Bmm_for_marg;

public :
    T* dev_ptr_H11;
    T* dev_ptr_H12;
    T* dev_ptr_H21;
    T* dev_ptr_H22;

    T* dev_ptr_B11;
    T* dev_ptr_B22;

public :
    T* dev_ptr_Hprior_eigenvec;
    T* dev_ptr_Hprior_eigenval;

public :
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior_eigenvec;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior_eigenval;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior_eigenval_sqrt;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior_eigenval_sqrt_inv;

public :
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hpp_for_marg;
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_Bpp_for_marg;

public :
    LevenbergLambda<T> lambda;

public :
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_dense_handle;

public :
    BlockRangeStreamSet block_range_stream_set;
    ProjTempStreamSet proj_temp_stream_set;
    JacobianResidualStreamSet jacobian_residual_stream_set;
    RobustInfoStreamSet robust_info_stream_set;
    HessianRHSStreamSet hessian_rhs_stream_set;
    UpdateStreamSet update_stream_set;

public :    // priors in host
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Jprior;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Eprior, host_Eprior_backup;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Bprior, host_Bprior_backup;

public :
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Hprior_temp;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Bprior_temp;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Jprior_temp;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> host_Eprior_temp;

public :
    T* dev_ptr_Hprior;
    T* dev_ptr_Bprior;
    T* dev_ptr_Jprior;
    T* dev_ptr_Eprior;

public :
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_states_init;

public :
    std::vector< Eigen::Matrix<T, 3, 1> > P_init_vec;
    std::vector< Eigen::Quaternion<T>   > Q_init_vec;
    std::vector< Eigen::Matrix<T, 3, 3> > R_init_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > V_init_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > BiasAcc_init_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > BiasGyr_init_vec;

public :
    std::vector< Eigen::Matrix<T, 3, 1> > P_vec;
    std::vector< Eigen::Quaternion<T>   > Q_vec;
    std::vector< Eigen::Matrix<T, 3, 3> > R_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > V_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > BiasAcc_vec;
    std::vector< Eigen::Matrix<T, 3, 1> > BiasGyr_vec;

public :
    std::set<int> curr_keyframe_idx_in_marg;
    std::set<int> prev_keyframe_idx_in_marg;

public : 
    bool have_prior;

public : 
    bool use_imu;
    bool is_stereo;
    bool pose_0_is_fixed;
    bool pose_ex_0_is_fixed;
    bool pose_ex_1_is_fixed;
    bool cur_td_is_fixed;

public :
    explicit SlidingWindow(int _num_key_frames, bool _decompose_prior);
    void Init();
    void ComputeShapes();
    bool AllocForStates();
    bool AllocForDeviceBigHessiansAndRHS();
    void MemcpyForCurrentFrame();
    bool MemcpyForStates();
    bool MemcpyForDeviceBigHessiansAndRHS();

    void ClearForCurrentFrame();
    bool ClearForStates();
    bool ClearForDeviceBigHessiansAndRHS();

    void InitCUDAHandles();
    void DestroyCUDAHandles() const;

    void SetStatesToInit();

    void LaunchProjFactorUpdateKernels();
    void LaunchIMUFactorUpdateKernels();

    void LaunchAllBlockRangeKernels();
    void LaunchProjBlockRangeKernelsForMarg();

    void DevJacobianSetZero() const;
    void DevJacobianSetZeroForMarg() const;
    void DevHessianAndRHSSetZero() const;
    void DevHessianAndRHSSetZeroForMarg() const;
    void LaunchJacResRobInfoKernels();

    void LaunchProjTempKernels();
    void LaunchJacobianAndResidualKernels();
    void LaunchRobustInfoKernels();

    void LaunchHessianAndRHSKernels();
    void LaunchHessianAndRHSKernelsForMarg();

    void ComputeLambdaInitLMV2();

    void LaunchSolveDeltaKernels();
    void BackupStatesAndInvDepth();
    void LaunchUpdateKernels();
    void LaunchAllFactorUpdateKernels();
    void LaunchStatesAddDeltaKernels() const;
    void LaunchInvDepthAddDeltaKernels() const;

    void BackupBpriorAndEprior();
    void UpdateBpriorAndEprior() const;
    void RollBackBpriorAndEprior();

    bool IsGoodStepInLM();

    void RollBackStatesAndInvDepth();

    void LaunchMarginalizeInvDepthKernels();
    void MargKeyFrameWithPriorDecomp();
    void MargKeyFrameWithoutPriorDecomp();
    void LaunchMarginalizeKeyFrameKernels();

    void AlignYawToFirstFrame();

    void SetPriorZero();

    void SetInitStates(const Eigen::MatrixXd& init_states);

    void PrintStates();

    bool NeedToMarginalize();
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_SLIDING_WINDOW_H
