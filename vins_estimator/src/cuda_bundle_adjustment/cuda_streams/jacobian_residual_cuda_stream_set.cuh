//
// Created by lmf on 23-7-26.
//

#ifndef CUDA_BA_JACOBIAN_RESIDUAL_CUDA_STREAM_SET_CUH
#define CUDA_BA_JACOBIAN_RESIDUAL_CUDA_STREAM_SET_CUH

#include <cuda_runtime.h>

namespace VINS_FUSION_CUDA_BA {

class JacobianResidualStreamSet {
public :
    cudaStream_t cuda_stream_proj_2f1c_jacobian_0_l{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_0_r{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_1_l{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_1_r{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_2_l{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_2_r{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_3{};
    cudaStream_t cuda_stream_proj_2f1c_jacobian_4{};
    cudaStream_t cuda_stream_proj_2f1c_residual{};

    cudaStream_t cuda_stream_proj_2f2c_jacobian_0_l{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_0_r{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_1_l{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_1_r{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_2_l{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_2_r{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_3_l{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_3_r{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_4{};
    cudaStream_t cuda_stream_proj_2f2c_jacobian_5{};
    cudaStream_t cuda_stream_proj_2f2c_residual{};

    cudaStream_t cuda_stream_proj_1f2c_jacobian_0_l{};
    cudaStream_t cuda_stream_proj_1f2c_jacobian_0_r{};
    cudaStream_t cuda_stream_proj_1f2c_jacobian_1_l{};
    cudaStream_t cuda_stream_proj_1f2c_jacobian_1_r{};
    cudaStream_t cuda_stream_proj_1f2c_jacobian_2{};
    cudaStream_t cuda_stream_proj_1f2c_jacobian_3{};
    cudaStream_t cuda_stream_proj_1f2c_residual{};

    cudaStream_t cuda_stream_imu_jacobian_0{};
    cudaStream_t cuda_stream_imu_jacobian_1{};
    cudaStream_t cuda_stream_imu_jacobian_2{};
    cudaStream_t cuda_stream_imu_jacobian_3{};
    cudaStream_t cuda_stream_imu_residual{};

public :
    JacobianResidualStreamSet();
    ~JacobianResidualStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_JACOBIAN_RESIDUAL_CUDA_STREAM_SET_CUH
