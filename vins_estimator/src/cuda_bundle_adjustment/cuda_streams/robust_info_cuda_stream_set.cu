//
// Created by lmf on 23-7-26.
//

#include "robust_info_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

RobustInfoStreamSet::RobustInfoStreamSet() {
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_robust_info, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_robust_info, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_robust_info, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_imu_robust_info, cudaStreamNonBlocking);
}

RobustInfoStreamSet::~RobustInfoStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(cuda_stream_proj_2f1c_robust_info);
    cudaStreamDestroy(cuda_stream_proj_2f2c_robust_info);
    cudaStreamDestroy(cuda_stream_proj_1f2c_robust_info);

    cudaStreamDestroy(cuda_stream_imu_robust_info);
}

void RobustInfoStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(cuda_stream_proj_2f1c_robust_info);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_robust_info);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_robust_info);

    cudaStreamSynchronize(cuda_stream_imu_robust_info);
}

} // namespace VINS_FUSION_CUDA_BA