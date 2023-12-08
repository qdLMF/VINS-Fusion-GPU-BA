//
// Created by lmf on 23-7-30.
//

#include "block_range_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

BlockRangeStreamSet::BlockRangeStreamSet() {
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu, cudaStreamNonBlocking);
}

BlockRangeStreamSet::~BlockRangeStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(cuda_stream_proj_2f1c);
    cudaStreamDestroy(cuda_stream_proj_2f2c);
    cudaStreamDestroy(cuda_stream_proj_1f2c);
    cudaStreamDestroy(cuda_stream_imu);
}

void BlockRangeStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(cuda_stream_proj_2f1c);
    cudaStreamSynchronize(cuda_stream_proj_2f2c);
    cudaStreamSynchronize(cuda_stream_proj_1f2c);
    cudaStreamSynchronize(cuda_stream_imu);
}

} // namespace VINS_FUSION_CUDA_BA


