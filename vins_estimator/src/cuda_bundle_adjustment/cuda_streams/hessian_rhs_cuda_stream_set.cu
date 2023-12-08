//
// Created by lmf on 23-7-26.
//

#include "hessian_rhs_cuda_stream_set.cuh"

namespace VINS_FUSION_CUDA_BA {

HessianRHSStreamSet::HessianRHSStreamSet() {
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_00, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_01, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_02, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_03, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_04, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_11, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_12, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_13, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_14, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_22, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_23, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_24, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_33, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_34, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_hessian_44, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_rhs_0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_rhs_1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_rhs_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_rhs_3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f1c_rhs_4, cudaStreamNonBlocking);

    // ----------

    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_00, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_01, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_02, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_03, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_04, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_05, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_11, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_12, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_13, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_14, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_15, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_22, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_23, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_24, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_25, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_33, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_34, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_35, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_44, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_45, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_hessian_55, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_4, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_2f2c_rhs_5, cudaStreamNonBlocking);

    // ----------

    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_00, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_01, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_02, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_03, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_11, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_12, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_13, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_22, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_23, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_hessian_33, cudaStreamNonBlocking);

    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_rhs_0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_rhs_1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_rhs_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_proj_1f2c_rhs_3, cudaStreamNonBlocking);

    // ----------

    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_00, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_01, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_02, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_03, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_11, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_12, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_13, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_22, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_23, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_hessian_33, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_rhs_0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_rhs_1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_rhs_2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream_imu_rhs_3, cudaStreamNonBlocking);
}

HessianRHSStreamSet::~HessianRHSStreamSet() {
    SyncAllStream();

    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_00);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_01);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_02);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_03);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_04);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_11);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_12);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_13);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_14);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_22);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_23);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_24);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_33);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_34);
    cudaStreamDestroy(cuda_stream_proj_2f1c_hessian_44);

    cudaStreamDestroy(cuda_stream_proj_2f1c_rhs_0);
    cudaStreamDestroy(cuda_stream_proj_2f1c_rhs_1);
    cudaStreamDestroy(cuda_stream_proj_2f1c_rhs_2);
    cudaStreamDestroy(cuda_stream_proj_2f1c_rhs_3);
    cudaStreamDestroy(cuda_stream_proj_2f1c_rhs_4);

    // ----------

    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_00);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_01);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_02);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_03);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_04);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_05);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_11);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_12);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_13);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_14);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_15);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_22);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_23);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_24);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_25);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_33);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_34);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_35);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_44);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_45);
    cudaStreamDestroy(cuda_stream_proj_2f2c_hessian_55);

    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_0);
    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_1);
    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_2);
    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_3);
    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_4);
    cudaStreamDestroy(cuda_stream_proj_2f2c_rhs_5);

    // ----------

    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_00);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_01);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_02);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_03);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_11);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_12);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_13);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_22);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_23);
    cudaStreamDestroy(cuda_stream_proj_1f2c_hessian_33);

    cudaStreamDestroy(cuda_stream_proj_1f2c_rhs_0);
    cudaStreamDestroy(cuda_stream_proj_1f2c_rhs_1);
    cudaStreamDestroy(cuda_stream_proj_1f2c_rhs_2);
    cudaStreamDestroy(cuda_stream_proj_1f2c_rhs_3);

    // ----------

    cudaStreamDestroy(cuda_stream_imu_hessian_00);
    cudaStreamDestroy(cuda_stream_imu_hessian_01);
    cudaStreamDestroy(cuda_stream_imu_hessian_02);
    cudaStreamDestroy(cuda_stream_imu_hessian_03);
    cudaStreamDestroy(cuda_stream_imu_hessian_11);
    cudaStreamDestroy(cuda_stream_imu_hessian_12);
    cudaStreamDestroy(cuda_stream_imu_hessian_13);
    cudaStreamDestroy(cuda_stream_imu_hessian_22);
    cudaStreamDestroy(cuda_stream_imu_hessian_23);
    cudaStreamDestroy(cuda_stream_imu_hessian_33);
    cudaStreamDestroy(cuda_stream_imu_rhs_0);
    cudaStreamDestroy(cuda_stream_imu_rhs_1);
    cudaStreamDestroy(cuda_stream_imu_rhs_2);
    cudaStreamDestroy(cuda_stream_imu_rhs_3);
}

void HessianRHSStreamSet::SyncAllStream() const {
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_00);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_01);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_02);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_03);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_04);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_11);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_12);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_13);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_14);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_22);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_23);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_24);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_33);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_34);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_hessian_44);

    cudaStreamSynchronize(cuda_stream_proj_2f1c_rhs_0);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_rhs_1);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_rhs_2);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_rhs_3);
    cudaStreamSynchronize(cuda_stream_proj_2f1c_rhs_4);

    // ----------

    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_00);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_01);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_02);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_03);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_04);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_05);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_11);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_12);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_13);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_14);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_15);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_22);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_23);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_24);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_25);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_33);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_34);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_35);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_44);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_45);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_hessian_55);

    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_0);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_1);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_2);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_3);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_4);
    cudaStreamSynchronize(cuda_stream_proj_2f2c_rhs_5);

    // ----------

    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_00);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_01);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_02);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_03);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_11);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_12);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_13);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_22);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_23);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_hessian_33);

    cudaStreamSynchronize(cuda_stream_proj_1f2c_rhs_0);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_rhs_1);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_rhs_2);
    cudaStreamSynchronize(cuda_stream_proj_1f2c_rhs_3);

    // ----------

    cudaStreamSynchronize(cuda_stream_imu_hessian_00);
    cudaStreamSynchronize(cuda_stream_imu_hessian_01);
    cudaStreamSynchronize(cuda_stream_imu_hessian_02);
    cudaStreamSynchronize(cuda_stream_imu_hessian_03);
    cudaStreamSynchronize(cuda_stream_imu_hessian_11);
    cudaStreamSynchronize(cuda_stream_imu_hessian_12);
    cudaStreamSynchronize(cuda_stream_imu_hessian_13);
    cudaStreamSynchronize(cuda_stream_imu_hessian_22);
    cudaStreamSynchronize(cuda_stream_imu_hessian_23);
    cudaStreamSynchronize(cuda_stream_imu_hessian_33);
    cudaStreamSynchronize(cuda_stream_imu_rhs_0);
    cudaStreamSynchronize(cuda_stream_imu_rhs_1);
    cudaStreamSynchronize(cuda_stream_imu_rhs_2);
    cudaStreamSynchronize(cuda_stream_imu_rhs_3);
}

} // namespace VINS_FUSION_CUDA_BA


