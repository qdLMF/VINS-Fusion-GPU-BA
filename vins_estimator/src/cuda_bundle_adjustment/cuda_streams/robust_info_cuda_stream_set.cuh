//
// Created by lmf on 23-7-26.
//

#ifndef CUDA_BA_ROBUST_INFO_CUDA_STREAM_SET_CUH
#define CUDA_BA_ROBUST_INFO_CUDA_STREAM_SET_CUH

namespace VINS_FUSION_CUDA_BA {

class RobustInfoStreamSet {
public :
    cudaStream_t cuda_stream_proj_2f1c_robust_info{};
    cudaStream_t cuda_stream_proj_2f2c_robust_info{};
    cudaStream_t cuda_stream_proj_1f2c_robust_info{};

    cudaStream_t cuda_stream_imu_robust_info{};

public :
    RobustInfoStreamSet();
    ~RobustInfoStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_ROBUST_INFO_CUDA_STREAM_SET_CUH
