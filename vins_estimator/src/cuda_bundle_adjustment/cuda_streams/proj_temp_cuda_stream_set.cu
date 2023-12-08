//
// Created by lmf on 23-7-31.
//

#include "proj_temp_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

ProjTempStreamSet::ProjTempStreamSet() {
    cudaStreamCreateWithFlags(&proj_2f1c_temp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&proj_2f2c_temp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&proj_1f2c_temp, cudaStreamNonBlocking);
}

ProjTempStreamSet::~ProjTempStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(proj_2f1c_temp);
    cudaStreamDestroy(proj_2f2c_temp);
    cudaStreamDestroy(proj_1f2c_temp);
}

void ProjTempStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(proj_2f1c_temp);
    cudaStreamSynchronize(proj_2f2c_temp);
    cudaStreamSynchronize(proj_1f2c_temp);
}

} // namespace VINS_FUSION_CUDA_BA

