//
// Created by lmf on 23-7-30.
//

#ifndef CUDA_BA_BLOCK_RANGE_CUDA_STREAM_SET_CUH
#define CUDA_BA_BLOCK_RANGE_CUDA_STREAM_SET_CUH

#include <cuda_runtime.h>

namespace VINS_FUSION_CUDA_BA {

class BlockRangeStreamSet {
public :
    cudaStream_t cuda_stream_proj_2f1c{};
    cudaStream_t cuda_stream_proj_2f2c{};
    cudaStream_t cuda_stream_proj_1f2c{};
    cudaStream_t cuda_stream_imu{};

public :
    BlockRangeStreamSet();
    ~BlockRangeStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_BLOCK_RANGE_CUDA_STREAM_SET_CUH
