//
// Created by lmf on 23-7-26.
//

#ifndef CUDA_BA_HESSIAN_RHS_CUDA_STREAM_SET_CUH
#define CUDA_BA_HESSIAN_RHS_CUDA_STREAM_SET_CUH

#include <cuda_runtime.h>

namespace VINS_FUSION_CUDA_BA {

class HessianRHSStreamSet {
public :
    cudaStream_t cuda_stream_proj_2f1c_hessian_00{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_01{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_02{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_03{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_04{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_11{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_12{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_13{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_14{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_22{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_23{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_24{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_33{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_34{};
    cudaStream_t cuda_stream_proj_2f1c_hessian_44{};

    cudaStream_t cuda_stream_proj_2f1c_rhs_0{};
    cudaStream_t cuda_stream_proj_2f1c_rhs_1{};
    cudaStream_t cuda_stream_proj_2f1c_rhs_2{};
    cudaStream_t cuda_stream_proj_2f1c_rhs_3{};
    cudaStream_t cuda_stream_proj_2f1c_rhs_4{};

    // ----------

    cudaStream_t cuda_stream_proj_2f2c_hessian_00{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_01{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_02{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_03{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_04{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_05{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_11{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_12{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_13{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_14{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_15{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_22{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_23{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_24{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_25{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_33{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_34{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_35{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_44{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_45{};
    cudaStream_t cuda_stream_proj_2f2c_hessian_55{};

    cudaStream_t cuda_stream_proj_2f2c_rhs_0{};
    cudaStream_t cuda_stream_proj_2f2c_rhs_1{};
    cudaStream_t cuda_stream_proj_2f2c_rhs_2{};
    cudaStream_t cuda_stream_proj_2f2c_rhs_3{};
    cudaStream_t cuda_stream_proj_2f2c_rhs_4{};
    cudaStream_t cuda_stream_proj_2f2c_rhs_5{};

    // ----------

    cudaStream_t cuda_stream_proj_1f2c_hessian_00{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_01{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_02{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_03{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_11{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_12{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_13{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_22{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_23{};
    cudaStream_t cuda_stream_proj_1f2c_hessian_33{};

    cudaStream_t cuda_stream_proj_1f2c_rhs_0{};
    cudaStream_t cuda_stream_proj_1f2c_rhs_1{};
    cudaStream_t cuda_stream_proj_1f2c_rhs_2{};
    cudaStream_t cuda_stream_proj_1f2c_rhs_3{};

    // ----------

    cudaStream_t cuda_stream_imu_hessian_00{};
    cudaStream_t cuda_stream_imu_hessian_01{};
    cudaStream_t cuda_stream_imu_hessian_02{};
    cudaStream_t cuda_stream_imu_hessian_03{};
    cudaStream_t cuda_stream_imu_hessian_11{};
    cudaStream_t cuda_stream_imu_hessian_12{};
    cudaStream_t cuda_stream_imu_hessian_13{};
    cudaStream_t cuda_stream_imu_hessian_22{};
    cudaStream_t cuda_stream_imu_hessian_23{};
    cudaStream_t cuda_stream_imu_hessian_33{};

    cudaStream_t cuda_stream_imu_rhs_0{};
    cudaStream_t cuda_stream_imu_rhs_1{};
    cudaStream_t cuda_stream_imu_rhs_2{};
    cudaStream_t cuda_stream_imu_rhs_3{};

public :
    HessianRHSStreamSet();
    ~HessianRHSStreamSet();
    void SyncAllStream() const;
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_HESSIAN_RHS_CUDA_STREAM_SET_CUH
