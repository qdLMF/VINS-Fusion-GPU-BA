//
// Created by lmf on 23-10-30.
//

#ifndef CUDA_BA_UPDATE_CUDA_STREAM_SET_CUH
#define CUDA_BA_UPDATE_CUDA_STREAM_SET_CUH

namespace VINS_FUSION_CUDA_BA {

class UpdateStreamSet {
public :
    cudaStream_t cuda_stream_proj_2f1c_update{};
    cudaStream_t cuda_stream_proj_2f2c_update{};
    cudaStream_t cuda_stream_proj_1f2c_update{};

    cudaStream_t cuda_stream_imu_update{};

public :
    UpdateStreamSet();
    ~UpdateStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_UPDATE_CUDA_STREAM_SET_CUH
