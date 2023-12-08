//
// Created by lmf on 23-7-26.
//

#include "jacobian_residual_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

JacobianResidualStreamSet::JacobianResidualStreamSet() {
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_0_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_0_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_1_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_1_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_2_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_2_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_jacobian_4, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_residual, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_0_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_0_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_1_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_1_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_2_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_2_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_3_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_3_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_4, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_jacobian_5, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_residual, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_0_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_0_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_1_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_1_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_jacobian_3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_residual, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_imu_jacobian_0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_jacobian_1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_jacobian_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_jacobian_3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_residual, cudaStreamNonBlocking);
}

JacobianResidualStreamSet::~JacobianResidualStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_0_l);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_0_r);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_1_l);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_1_r);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_2_l);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_2_r);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_3);
    cudaStreamDestroy(cuda_stream_proj_2f1c_jacobian_4);
    cudaStreamDestroy(cuda_stream_proj_2f1c_residual);

    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_0_l);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_0_r);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_1_l);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_1_r);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_2_l);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_2_r);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_3_l);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_3_r);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_4);
    cudaStreamDestroy(cuda_stream_proj_2f2c_jacobian_5);
    cudaStreamDestroy(cuda_stream_proj_2f2c_residual);

    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_0_l);
    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_0_r);
    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_1_l);
    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_1_r);
    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_2);
    cudaStreamDestroy(cuda_stream_proj_1f2c_jacobian_3);
    cudaStreamDestroy(cuda_stream_proj_1f2c_residual);

    cudaStreamDestroy(cuda_stream_imu_jacobian_0);
    cudaStreamDestroy(cuda_stream_imu_jacobian_1);
    cudaStreamDestroy(cuda_stream_imu_jacobian_2);
    cudaStreamDestroy(cuda_stream_imu_jacobian_3);
    cudaStreamDestroy(cuda_stream_imu_residual);
}

void JacobianResidualStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_0_l);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_0_r);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_1_l);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_1_r);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_2_l);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_2_r);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_3);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_jacobian_4);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_residual);

    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_0_l);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_0_r);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_1_l);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_1_r);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_2_l);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_2_r);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_3_l);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_3_r);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_4);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_jacobian_5);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_residual);

    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_0_l);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_0_r);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_1_l);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_1_r);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_2);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_jacobian_3);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_residual);

    cudaStreamSynchronize(cuda_stream_imu_jacobian_0);
    cudaStreamSynchronize(cuda_stream_imu_jacobian_1);
    cudaStreamSynchronize(cuda_stream_imu_jacobian_2);
    cudaStreamSynchronize(cuda_stream_imu_jacobian_3);
    cudaStreamSynchronize(cuda_stream_imu_residual);
}

} // namespace VINS_FUSION_CUDA_BA

