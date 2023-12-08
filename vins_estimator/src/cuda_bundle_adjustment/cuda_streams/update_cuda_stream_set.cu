//
// Created by lmf on 23-10-30.
//

#include "update_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

UpdateStreamSet::UpdateStreamSet() {
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_update, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_update, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_update, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_imu_update, cudaStreamNonBlocking);
}

UpdateStreamSet::~UpdateStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(cuda_stream_proj_2f1c_update);
    cudaStreamDestroy(cuda_stream_proj_2f2c_update);
    cudaStreamDestroy(cuda_stream_proj_1f2c_update);

    cudaStreamDestroy(cuda_stream_imu_update);
}

void UpdateStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(cuda_stream_proj_2f1c_update);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_update);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_update);

    cudaStreamSynchronize(cuda_stream_imu_update);
}

} // namespace VINS_FUSION_CUDA_BA

