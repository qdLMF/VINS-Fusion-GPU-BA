//
// Created by lmf on 23-7-27.
//

#include "launch_kernels.cuh"

#define NUM_THREADS_PER_BLOCK_PRJ_UPD 16
#define NUM_THREADS_PER_BLOCK_PRJ_RHS 16
#define NUM_THREADS_PER_BLOCK_PRJ_HES 16
#define NUM_THREADS_PER_BLOCK_PRJ_RIF 16
#define NUM_THREADS_PER_BLOCK_PRJ_RES 16
#define NUM_THREADS_PER_BLOCK_PRJ_JAC 16
#define NUM_THREADS_PER_BLOCK_PRJ_TMP 16
#define NUM_THREADS_PER_BLOCK_PRJ_BRG 16

#define NUM_THREADS_PER_BLOCK_IMU_UPD 20
#define NUM_THREADS_PER_BLOCK_IMU_RHS 20
#define NUM_THREADS_PER_BLOCK_IMU_HES 20
#define NUM_THREADS_PER_BLOCK_IMU_RIF 20
#define NUM_THREADS_PER_BLOCK_IMU_RES 20
#define NUM_THREADS_PER_BLOCK_IMU_JAC 20
#define NUM_THREADS_PER_BLOCK_IMU_BRG 20

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void AllBlockRangeKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool use_imu,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_block_range
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_2f1c >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_block_range
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_2f2c >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_robust_info
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_1f2c >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_block_range
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_BRG;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0 , block_range_cuda_stream.cuda_stream_imu >>>(
            num_imu_factors,
            imu_dev_ptr_set
        );
    }

    // ----------------------------------------------------------------------------------------------------

    block_range_cuda_stream.SyncAllStream();
}
// instantiation
template void AllBlockRangeKernels<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool use_imu,
    bool is_stereo
);
template void AllBlockRangeKernels<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool use_imu,
    bool is_stereo
);

// ----------

template<typename T>
void AllProjTempKernels(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_temp
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_2f1c_temp >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_temp
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_2f2c_temp >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_temp
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_1f2c_temp >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    proj_temp_cuda_stream.SyncAllStream();
}
// instantiation
template void AllProjTempKernels<double>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);
template void AllProjTempKernels<float>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void AllJacobianAndResidualKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_factor_jacobian
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_0_l >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
        proj_2f1c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_0_r >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
        proj_2f1c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_1_l >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
        proj_2f1c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_1_r >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f1c_jacobian_2_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_2_l >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                false
            );
            proj_2f1c_jacobian_2_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_2_r >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                false
            );
        }
        proj_2f1c_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_3 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f1c_jacobian_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_4 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                false
            );
        }
    }
    // proj_2f1c_factor_residual
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_residual >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_factor_jacobian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_0_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
        proj_2f2c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_0_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
        proj_2f2c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_1_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
        proj_2f2c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_1_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f2c_jacobian_2_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_2_l >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                false
            );
            proj_2f2c_jacobian_2_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_2_r >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_2f2c_jacobian_3_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_3_l >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                false
            );
            proj_2f2c_jacobian_3_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_3_r >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                false
            );
        }
        proj_2f2c_jacobian_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_4 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f2c_jacobian_5<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_5 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                false
            );
        }
    }
    // proj_2f2c_factor_residual
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_residual >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_factor_jacobian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        if(!pose_ex_0_is_fixed) {
            proj_1f2c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_0_l >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                false
            );
            proj_1f2c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_0_r >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_1f2c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_1_l >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                false
            );
            proj_1f2c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_1_r >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                false
            );
        }
        proj_1f2c_jacobian_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_2 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            false
        );
        if(!cur_td_is_fixed) {
            proj_1f2c_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_3 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                false
            );
        }
    }
    // proj_1f2c_factor_residual
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_residual >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_factor_jacobian
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_JAC;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_jacobian_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_0 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
        imu_jacobian_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_1 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
        imu_jacobian_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_2 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
        imu_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_3 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
    }
    // imu_factor_residual
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_RES;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_residual >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    jacobian_residual_cuda_stream.SyncAllStream();
}
// instantiation
template void AllJacobianAndResidualKernels<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);
template void AllJacobianAndResidualKernels<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);

// ----------

template<typename T>
void AllRobustInfoKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_robust_info
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_2f1c_robust_info >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_robust_info
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_2f2c_robust_info >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_robust_info
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_1f2c_robust_info >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_robust_info
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_RIF;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0 , robust_info_cuda_stream.cuda_stream_imu_robust_info >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    robust_info_cuda_stream.SyncAllStream();
}
// instantiation
template void AllRobustInfoKernels<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);
template void AllRobustInfoKernels<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);

// ----------

template<typename T>
void AllHessianAndRHSKernels(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    T* Hpp,
    int leading_dim_Hpp,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    T* Hmm_diag,
    T* Bpp,
    T* Bmm,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_hessian
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_00 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        proj_2f1c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_01 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f1c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_02 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f1c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_03 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f1c_hessian_04<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_04 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f1c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_11 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f1c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_12 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f1c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_13 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f1c_hessian_14<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_14 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        if(!pose_ex_0_is_fixed) {
            proj_2f1c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_22 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
            proj_2f1c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_23 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
            if(!cur_td_is_fixed) {
                proj_2f1c_hessian_24<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_24 >>>(
                    num_proj_2f1c_factors,
                    proj_2f1c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
        }
        proj_2f1c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_33 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hmm_diag,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f1c_hessian_34<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_34 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
        }
        if(!cur_td_is_fixed) {
            proj_2f1c_hessian_44<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_44 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
    }
    // proj_2f1c_rhs
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_0 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            false
        );
        proj_2f1c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_1 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f1c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_2 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Bpp,
                false
            );
        }
        proj_2f1c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_3 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bmm,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f1c_rhs_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_4 >>>(
                num_proj_2f1c_factors,
                proj_2f1c_dev_ptr_set,
                Bpp,
                false
            );
        }
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_hessian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_00 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        proj_2f2c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_01 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f2c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_02 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_2f2c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_03 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f2c_hessian_04<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_04 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f2c_hessian_05<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_05 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f2c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_11 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f2c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_12 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_2f2c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_13 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        proj_2f2c_hessian_14<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_14 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f2c_hessian_15<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_15 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
        if(!pose_ex_0_is_fixed) {
            proj_2f2c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_22 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
            if(!pose_ex_1_is_fixed) {
                proj_2f2c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_23 >>>(
                    num_proj_2f2c_factors,
                    proj_2f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
            proj_2f2c_hessian_24<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_24 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
            if(!cur_td_is_fixed) {
                proj_2f2c_hessian_25<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_25 >>>(
                    num_proj_2f2c_factors,
                    proj_2f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
        }
        if(!pose_ex_1_is_fixed) {
            proj_2f2c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_33 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
            proj_2f2c_hessian_34<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_34 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
            if(!cur_td_is_fixed) {
                proj_2f2c_hessian_35<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_35 >>>(
                    num_proj_2f2c_factors,
                    proj_2f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
        }
        proj_2f2c_hessian_44<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_44 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hmm_diag,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f2c_hessian_45<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_45 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
        }
        if(!cur_td_is_fixed) {
            proj_2f2c_hessian_55<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_55 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
    }
    // proj_2f2c_rhs
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_0 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            false
        );
        proj_2f2c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_1 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            false
        );
        if(!pose_ex_0_is_fixed) {
            proj_2f2c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_2 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_2f2c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_3 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
        proj_2f2c_rhs_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_4 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bmm,
            false
        );
        if(!cur_td_is_fixed) {
            proj_2f2c_rhs_5<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_5 >>>(
                num_proj_2f2c_factors,
                proj_2f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_hessian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        if(!pose_ex_0_is_fixed) {
            proj_1f2c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_00 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
            if(!pose_ex_1_is_fixed) {
                proj_1f2c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_01 >>>(
                    num_proj_1f2c_factors,
                    proj_1f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
            proj_1f2c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_02 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
            if(!cur_td_is_fixed) {
                proj_1f2c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_03 >>>(
                    num_proj_1f2c_factors,
                    proj_1f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
        }
        if(!pose_ex_1_is_fixed) {
            proj_1f2c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_11 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
            proj_1f2c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_12 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
            if(!cur_td_is_fixed) {
                proj_1f2c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_13 >>>(
                    num_proj_1f2c_factors,
                    proj_1f2c_dev_ptr_set,
                    Hpp,
                    leading_dim_Hpp,
                    false
                );
            }
        }
        proj_1f2c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_22 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hmm_diag,
            false
        );
        if(!cur_td_is_fixed) {
            proj_1f2c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_23 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpm,
                leading_dim_Hpm,
                Hmp,
                leading_dim_Hmp,
                false
            );
        }
        if(!cur_td_is_fixed) {
            proj_1f2c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_33 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Hpp,
                leading_dim_Hpp,
                false
            );
        }
    }
    // proj_1f2c_rhs
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        if(!pose_ex_0_is_fixed) {
            proj_1f2c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_0 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
        if(!pose_ex_1_is_fixed) {
            proj_1f2c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_1 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
        proj_1f2c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_2 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Bmm,
            false
        );
        if(!cur_td_is_fixed) {
            proj_1f2c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_3 >>>(
                num_proj_1f2c_factors,
                proj_1f2c_dev_ptr_set,
                Bpp,
                false
            );
        }
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_factor_hessian
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_HES;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_00 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_01 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_02 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_03 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_11 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_12 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_13 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_22 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_23 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
        imu_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_33 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            false
        );
    }
    // imu_factor_rhs
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_HES;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_0 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            false
        );
        imu_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_1 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            false
        );
        imu_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_2 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            false
        );
        imu_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_3 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            false
        );
    }

    // ----------------------------------------------------------------------------------------------------

    hessian_rhs_cuda_stream.SyncAllStream();
}
// instantiation
template void AllHessianAndRHSKernels<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    double* Hpp,
    int leading_dim_Hpp,
    double* Hpm,
    int leading_dim_Hpm,
    double* Hmp,
    int leading_dim_Hmp,
    double* Hmm_diag,
    double* Bpp,
    double* Bmm,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);
template void AllHessianAndRHSKernels<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    float* Hpp,
    int leading_dim_Hpp,
    float* Hpm,
    int leading_dim_Hpm,
    float* Hmp,
    int leading_dim_Hmp,
    float* Hmm_diag,
    float* Bpp,
    float* Bmm,
    bool use_imu,
    bool is_stereo,
    bool pose_ex_0_is_fixed,
    bool pose_ex_1_is_fixed,
    bool cur_td_is_fixed
);

// ----------

template<typename T>
void AllUpdateKernels(
    const T* input_ex_para_0,
    const T* input_ex_para_1,
    const T* input_states,
    const T* input_inv_depth,
    const T* input_cur_td,
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    UpdateStreamSet& update_cuda_stream
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_update
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_UPD;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_update<T><<< num_blocks_local, num_threads_per_block_local, 0, update_cuda_stream.cuda_stream_proj_2f1c_update >>>(
            num_proj_2f1c_factors,
            input_ex_para_0,
            input_states,
            input_inv_depth,
            input_cur_td,
            proj_2f1c_dev_ptr_set
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_update
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_UPD;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_update<T><<< num_blocks_local, num_threads_per_block_local, 0, update_cuda_stream.cuda_stream_proj_2f2c_update >>>(
            num_proj_2f2c_factors,
            input_ex_para_0,
            input_ex_para_1,
            input_states,
            input_inv_depth,
            input_cur_td,
            proj_2f2c_dev_ptr_set
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_update
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_UPD;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_update<T><<< num_blocks_local, num_threads_per_block_local, 0, update_cuda_stream.cuda_stream_proj_1f2c_update >>>(
            num_proj_1f2c_factors,
            input_ex_para_0,
            input_ex_para_1,
            input_inv_depth,
            input_cur_td,
            proj_1f2c_dev_ptr_set
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_update
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_UPD;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_update<T><<< num_blocks_local, num_threads_per_block_local, 0 , update_cuda_stream.cuda_stream_imu_update >>>(
            num_imu_factors,
            input_states,
            imu_dev_ptr_set
        );
    }

    // ----------------------------------------------------------------------------------------------------

    update_cuda_stream.SyncAllStream();
}
// instantiation
template void AllUpdateKernels<double>(
    const double* input_ex_para_0,
    const double* input_ex_para_1,
    const double* input_states,
    const double* input_inv_depth,
    const double* input_cur_td,
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    UpdateStreamSet& update_cuda_stream
);
template void AllUpdateKernels<float>(
    const float* input_ex_para_0,
    const float* input_ex_para_1,
    const float* input_states,
    const float* input_inv_depth,
    const float* input_cur_td,
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    UpdateStreamSet& update_cuda_stream
);

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
void AllProjBlockRangeKernelsForMarg(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_block_range
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_2f1c >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_block_range
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_2f2c >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_block_range
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_BRG;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_block_range<T><<< num_blocks_local, num_threads_per_block_local, 0, block_range_cuda_stream.cuda_stream_proj_1f2c >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    block_range_cuda_stream.SyncAllStream();
}
// instantiation
template void AllProjBlockRangeKernelsForMarg<double>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool is_stereo
);
template void AllProjBlockRangeKernelsForMarg<float>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    BlockRangeStreamSet& block_range_cuda_stream,
    bool is_stereo
);

// ----------

template<typename T>
void AllProjTempKernelsForMarg(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_temp
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_2f1c_temp >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_temp
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_2f2c_temp >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_temp
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_TMP;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_temp<T><<< num_blocks_local, num_threads_per_block_local, 0, proj_temp_cuda_stream.proj_1f2c_temp >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    proj_temp_cuda_stream.SyncAllStream();
}
// instantiation
template void AllProjTempKernelsForMarg<double>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);
template void AllProjTempKernelsForMarg<float>(
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    ProjTempStreamSet& proj_temp_cuda_stream,
    bool is_stereo
);

// ----------

template<typename T>
void AllJacobianAndResidualKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_factor_jacobian
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_0_l >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_0_r >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_1_l >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_1_r >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_2_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_2_l >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_2_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_2_r >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_3 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
        proj_2f1c_jacobian_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_jacobian_4 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
    }
    // proj_2f1c_factor_residual
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f1c_residual >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_factor_jacobian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_0_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_0_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_1_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_1_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_2_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_2_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_2_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_2_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_3_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_3_l >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_3_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_3_r >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_4 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
        proj_2f2c_jacobian_5<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_jacobian_5 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
    }
    // proj_2f2c_factor_residual
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_2f2c_residual >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_factor_jacobian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_JAC;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_jacobian_0_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_0_l >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
        proj_1f2c_jacobian_0_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_0_l >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
        proj_1f2c_jacobian_1_l<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_1_l >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
        proj_1f2c_jacobian_1_r<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_1_l >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
        proj_1f2c_jacobian_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_2 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
        proj_1f2c_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_jacobian_3 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
    }
    // proj_1f2c_factor_residual
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RES;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_proj_1f2c_residual >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_factor_jacobian
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_JAC;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_jacobian_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_0 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
        imu_jacobian_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_1 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
        imu_jacobian_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_2 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
        imu_jacobian_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_jacobian_3 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
    }
    // imu_factor_residual
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_RES;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_residual<T><<< num_blocks_local, num_threads_per_block_local, 0 , jacobian_residual_cuda_stream.cuda_stream_imu_residual >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    jacobian_residual_cuda_stream.SyncAllStream();
}
// instantiation
template void AllJacobianAndResidualKernelsForMarg<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo
);
template void AllJacobianAndResidualKernelsForMarg<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    JacobianResidualStreamSet& jacobian_residual_cuda_stream,
    bool use_imu,
    bool is_stereo
);

// ----------

template<typename T>
void AllRobustInfoKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_robust_info
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_2f1c_robust_info >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_robust_info
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_2f2c_robust_info >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_robust_info
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RIF;  // 1;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0, robust_info_cuda_stream.cuda_stream_proj_1f2c_robust_info >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_robust_info
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_RIF;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_robust_info<T><<< num_blocks_local, num_threads_per_block_local, 0 , robust_info_cuda_stream.cuda_stream_imu_robust_info >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    robust_info_cuda_stream.SyncAllStream();
}
// instantiation
template void AllRobustInfoKernelsForMarg<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);
template void AllRobustInfoKernelsForMarg<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    RobustInfoStreamSet& robust_info_cuda_stream,
    bool use_imu,
    bool is_stereo
);

// ----------

template<typename T>
void AllHessianAndRHSKernelsForMarg(
    int num_imu_factors,
    IMUFactorDevPtrSet<T>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<T>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<T>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<T>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    T* Hpp,
    int leading_dim_Hpp,
    T* Hpm,
    int leading_dim_Hpm,
    T* Hmp,
    int leading_dim_Hmp,
    T* Hmm_diag,
    T* Bpp,
    T* Bmm,
    bool use_imu,
    bool is_stereo
) {
    cudaError_t cuda_status = cudaStreamSynchronize(0);
    assert(cuda_status == cudaSuccess);

    // ----------------------------------------------------------------------------------------------------

    // proj_2f1c_hessian
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_00 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_01 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_02 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_03 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f1c_hessian_04<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_04 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_11 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_12 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_13 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f1c_hessian_14<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_14 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_22 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_23 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f1c_hessian_24<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_24 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f1c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_33 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hmm_diag,
            true
        );
        proj_2f1c_hessian_34<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_34 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f1c_hessian_44<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_hessian_44 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
    }
    // proj_2f1c_rhs
    {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_2f1c_factors / _num_threads_local;
        if( (num_proj_2f1c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f1c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_0 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f1c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_1 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f1c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_2 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f1c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_3 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bmm,
            true
        );
        proj_2f1c_rhs_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f1c_rhs_4 >>>(
            num_proj_2f1c_factors,
            proj_2f1c_dev_ptr_set,
            Bpp,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_2f2c_hessian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_00 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_01 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_02 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_03 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_04<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_04 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f2c_hessian_05<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_05 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_11 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_12 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_13 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_14<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_14 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f2c_hessian_15<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_15 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_22 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_23 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_24<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_24 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f2c_hessian_25<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_25 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_33 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_34<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_34 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f2c_hessian_35<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_35 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_2f2c_hessian_44<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_44 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hmm_diag,
            true
        );
        proj_2f2c_hessian_45<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_45 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_2f2c_hessian_55<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_hessian_55 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
    }
    // proj_2f2c_rhs
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_2f2c_factors / _num_threads_local;
        if( (num_proj_2f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_2f2c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_0 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f2c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_1 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f2c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_2 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f2c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_3 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_2f2c_rhs_4<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_4 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bmm,
            true
        );
        proj_2f2c_rhs_5<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_2f2c_rhs_5 >>>(
            num_proj_2f2c_factors,
            proj_2f2c_dev_ptr_set,
            Bpp,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // proj_1f2c_hessian
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_HES;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_00 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_1f2c_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_01 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_1f2c_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_02 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_1f2c_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_03 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_1f2c_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_11 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_1f2c_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_12 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_1f2c_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_13 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        proj_1f2c_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_22 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hmm_diag,
            true
        );
        proj_1f2c_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_23 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpm,
            leading_dim_Hpm,
            Hmp,
            leading_dim_Hmp,
            true
        );
        proj_1f2c_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_hessian_33 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
    }
    // proj_1f2c_rhs
    if(is_stereo) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_PRJ_RHS;
        int _num_blocks_local = num_proj_1f2c_factors / _num_threads_local;
        if( (num_proj_1f2c_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        proj_1f2c_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_0 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_1f2c_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_1 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Bpp,
            true
        );
        proj_1f2c_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_2 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Bmm,
            true
        );
        proj_1f2c_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_proj_1f2c_rhs_3 >>>(
            num_proj_1f2c_factors,
            proj_1f2c_dev_ptr_set,
            Bpp,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    // imu_factor_hessian
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_HES;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_hessian_00<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_00 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_01<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_01 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_02<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_02 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_03<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_03 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_11<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_11 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_12<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_12 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_13<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_13 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_22<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_22 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_23<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_23 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
        imu_hessian_33<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_hessian_33 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Hpp,
            leading_dim_Hpp,
            true
        );
    }
    // imu_factor_rhs
    if(use_imu) {
        int _num_threads_local = NUM_THREADS_PER_BLOCK_IMU_RHS;
        int _num_blocks_local = num_imu_factors / _num_threads_local;
        if( (num_imu_factors % _num_threads_local) != 0 )
            _num_blocks_local += 1;
        dim3 num_blocks_local(_num_blocks_local);
        dim3 num_threads_per_block_local(_num_threads_local);

        imu_rhs_0<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_0 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            true
        );
        imu_rhs_1<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_1 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            true
        );
        imu_rhs_2<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_2 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            true
        );
        imu_rhs_3<T><<< num_blocks_local, num_threads_per_block_local, 0 , hessian_rhs_cuda_stream.cuda_stream_imu_rhs_3 >>>(
            num_imu_factors,
            imu_dev_ptr_set,
            Bpp,
            true
        );
    }

    // ----------------------------------------------------------------------------------------------------

    hessian_rhs_cuda_stream.SyncAllStream();
}
// instantiation
template void AllHessianAndRHSKernelsForMarg<double>(
    int num_imu_factors,
    IMUFactorDevPtrSet<double>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<double>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<double>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<double>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    double* Hpp,
    int leading_dim_Hpp,
    double* Hpm,
    int leading_dim_Hpm,
    double* Hmp,
    int leading_dim_Hmp,
    double* Hmm_diag,
    double* Bpp,
    double* Bmm,
    bool use_imu,
    bool is_stereo
);
template void AllHessianAndRHSKernelsForMarg<float>(
    int num_imu_factors,
    IMUFactorDevPtrSet<float>& imu_dev_ptr_set,
    int num_proj_2f1c_factors,
    Proj2F1CFactorDevPtrSet<float>& proj_2f1c_dev_ptr_set,
    int num_proj_2f2c_factors,
    Proj2F2CFactorDevPtrSet<float>& proj_2f2c_dev_ptr_set,
    int num_proj_1f2c_factors,
    Proj1F2CFactorDevPtrSet<float>& proj_1f2c_dev_ptr_set,
    HessianRHSStreamSet& hessian_rhs_cuda_stream,
    float* Hpp,
    int leading_dim_Hpp,
    float* Hpm,
    int leading_dim_Hpm,
    float* Hmp,
    int leading_dim_Hmp,
    float* Hmm_diag,
    float* Bpp,
    float* Bmm,
    bool use_imu,
    bool is_stereo
);

} // namespace VINS_FUSION_CUDA_BA
