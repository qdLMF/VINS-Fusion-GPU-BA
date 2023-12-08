//
// Created by lmf on 23-7-31.
//

#ifndef CUDA_BA_PROJ_TEMP_CUDA_STREAM_SET_CUH
#define CUDA_BA_PROJ_TEMP_CUDA_STREAM_SET_CUH

namespace VINS_FUSION_CUDA_BA {

class ProjTempStreamSet {
public :
    cudaStream_t proj_2f1c_temp{};
    cudaStream_t proj_2f2c_temp{};
    cudaStream_t proj_1f2c_temp{};

public :
    ProjTempStreamSet();
    ~ProjTempStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_PROJ_TEMP_CUDA_STREAM_SET_CUH
