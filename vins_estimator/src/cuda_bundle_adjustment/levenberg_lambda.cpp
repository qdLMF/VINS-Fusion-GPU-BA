//
// Created by lmf on 23-7-31.
//

#include "levenberg_lambda.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
void LevenbergLambda<T>::Reset() {
    currentLambda = 0.0;
    stopThresholdLM = 0.0;
    currentChi = 0.0;
    maxDiagonal = 0;
    tau = 1e-16;
    ni = 2;
    eta = 1e-8;
    num_consecutive_bad_steps = 0;
}

template<typename T>
void LevenbergLambda<T>::ComputeLambdaInitLMV2() {
    Reset();

    host_imu_chi2.resize(num_imu_factors);
    host_proj_2f1c_chi2.resize(num_proj_2f1c_factors);
    host_proj_2f2c_chi2.resize(num_proj_2f2c_factors);
    host_proj_1f2c_chi2.resize(num_proj_1f2c_factors);
    host_err_prior.resize(shapes->num_elem_Bpp);

    cudaMemcpy(host_imu_chi2.data(), imu_dev_ptr->robust_chi2, num_imu_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_2f1c_chi2.data(), proj_2f1c_dev_ptr->robust_chi2, num_proj_2f1c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_2f2c_chi2.data(), proj_2f2c_dev_ptr->robust_chi2, num_proj_2f2c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_1f2c_chi2.data(), proj_1f2c_dev_ptr->robust_chi2, num_proj_1f2c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_err_prior.data(), dev_ptr_err_prior, shapes->num_elem_Bpp * 1 * sizeof(T), cudaMemcpyDeviceToHost);

    // currentChi = 0.5 * (host_imu_chi2.sum() + host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum() + host_err_prior.squaredNorm());
    currentChi = 0.5 * (host_imu_chi2.sum() + host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum());
    stopThresholdLM = 1e-10 * currentChi;

    unsigned int nrows = shapes->num_rows_Hpp;
    unsigned int ncols = shapes->num_cols_Hpp;
    host_Hpp.resize(nrows, ncols);
    cudaMemcpy(host_Hpp.data(), dev_ptr_Hpp, sizeof(T) * nrows * ncols, cudaMemcpyDeviceToHost);

    unsigned int nelem = shapes->num_elem_Hmm_diag;
    host_Hmm_diag.resize(nelem);
    cudaMemcpy(host_Hmm_diag.data(), dev_ptr_Hmm_diag, sizeof(T) * nelem, cudaMemcpyDeviceToHost);

    maxDiagonal = 0;
    for (int i = 0; i < host_Hpp.cols(); i++) {
        maxDiagonal = std::max(T(fabs(host_Hpp(i, i))), maxDiagonal);
    }
    for (int i = 0; i < host_Hmm_diag.size(); i++) {
        maxDiagonal = std::max(T(fabs(host_Hmm_diag(i))), maxDiagonal);
    }

    // printf("---------------------------------------- \n");
    // printf("maxDiagonal   : %.20f \n", maxDiagonal);
    // printf("currentChi    : %.20f \n", currentChi);

    maxDiagonal = std::min(T(5e10), maxDiagonal);   // 5e10

    currentLambda = tau * maxDiagonal;

    // printf("currentChi    : %.20f \n", currentChi);
    // printf("currentLambda : %.20f \n", currentLambda);
    // std::cout << "host_err_prior      : " << host_err_prior.squaredNorm() / weight << std::endl;
    // std::cout << "host_imu_chi2       : " << host_imu_chi2.sum() << std::endl;
    // std::cout << "host_proj_2f1c_chi2 : " << host_proj_2f1c_chi2.sum() << std::endl;
    // std::cout << "host_proj_2f2c_chi2 : " << host_proj_2f2c_chi2.sum() << std::endl;
    // std::cout << "host_proj_1f2c_chi2 : " << host_proj_1f2c_chi2.sum() << std::endl;
    // printf("---------------------------------------- \n");

    cudaMemcpy(dev_ptr_lambda, &currentLambda, 1 * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
bool LevenbergLambda<T>::IsGoodStepInLM() {
    host_delta_ex_para_0.resize(6);
    host_delta_ex_para_1.resize(6);
    host_delta_states.resize(num_key_frames * 15);
    host_delta_inv_depth.resize(num_world_points);
    host_delta_cur_td.resize(1);

    cudaMemcpy(host_delta_ex_para_0.data(), dev_ptr_delta_ex_para_0, 6 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_ex_para_1.data(), dev_ptr_delta_ex_para_1, 6 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_states.data(), dev_ptr_delta_states, (num_key_frames * 15) * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_inv_depth.data(), dev_ptr_delta_inv_depth, num_world_points * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_cur_td.data(), dev_ptr_delta_cur_td, 1 * 1 * sizeof(T), cudaMemcpyDeviceToHost);

    host_Bpp.resize(shapes->num_elem_Bpp);
    host_Bmm.resize(shapes->num_elem_Bmm);
    cudaMemcpy(host_Bpp.data(), dev_ptr_Bpp, shapes->num_elem_Bpp * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Bmm.data(), dev_ptr_Bmm, shapes->num_elem_Bmm * sizeof(T), cudaMemcpyDeviceToHost);

    host_Bprior.resize(shapes->num_elem_Bpp);
    cudaMemcpy(host_Bprior.data(), dev_ptr_Bprior, shapes->num_elem_Bpp * sizeof(T), cudaMemcpyDeviceToHost);
    host_Bpp -= host_Bprior;

    host_imu_chi2.resize(num_imu_factors);
    host_proj_2f1c_chi2.resize(num_proj_2f1c_factors);
    host_proj_2f2c_chi2.resize(num_proj_2f2c_factors);
    host_proj_1f2c_chi2.resize(num_proj_1f2c_factors);
    host_err_prior.resize(shapes->num_elem_Bpp);

    cudaMemcpy(host_imu_chi2.data(), imu_dev_ptr->robust_chi2, num_imu_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_2f1c_chi2.data(), proj_2f1c_dev_ptr->robust_chi2, num_proj_2f1c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_2f2c_chi2.data(), proj_2f2c_dev_ptr->robust_chi2, num_proj_2f2c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_proj_1f2c_chi2.data(), proj_1f2c_dev_ptr->robust_chi2, num_proj_1f2c_factors * 1 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_err_prior.data(), dev_ptr_err_prior, shapes->num_elem_Bpp * 1 * sizeof(T), cudaMemcpyDeviceToHost);

    Eigen::Matrix<T, Eigen::Dynamic, 1> delta_x;
    delta_x.resize(host_delta_ex_para_0.size() + host_delta_ex_para_1.size() + host_delta_cur_td.size() + host_delta_states.size() + host_delta_inv_depth.size());
    delta_x.segment(0, host_delta_ex_para_0.size()) = host_delta_ex_para_0;
    delta_x.segment(host_delta_ex_para_0.size(), host_delta_ex_para_1.size()) = host_delta_ex_para_1;
    delta_x.segment(host_delta_ex_para_0.size() + host_delta_ex_para_1.size(), host_delta_cur_td.size()) = host_delta_cur_td;
    delta_x.segment(host_delta_ex_para_0.size() + host_delta_ex_para_1.size() + host_delta_cur_td.size(), host_delta_states.size()) = host_delta_states;
    delta_x.segment(host_delta_ex_para_0.size() + host_delta_ex_para_1.size() + host_delta_cur_td.size() + host_delta_states.size(), host_delta_inv_depth.size()) = host_delta_inv_depth;
    Eigen::Matrix<T, Eigen::Dynamic, 1> b;
    b.resize(host_Bpp.size() + host_Bmm.size());
    b.segment(0, host_Bpp.size()) = host_Bpp;
    b.segment(host_Bpp.size(), host_Bmm.size()) = host_Bmm;

    unsigned int nrows = shapes->num_rows_Hpp;
    unsigned int ncols = shapes->num_cols_Hpp;
    host_Hpp.resize(nrows, ncols);
    cudaMemcpy(host_Hpp.data(), dev_ptr_Hpp_with_lambda, sizeof(T) * nrows * ncols, cudaMemcpyDeviceToHost);
    unsigned int nelem = shapes->num_elem_Hmm_diag;
    host_Hmm_diag.resize(nelem);
    cudaMemcpy(host_Hmm_diag.data(), dev_ptr_Hmm_diag_with_lambda, sizeof(T) * nelem, cudaMemcpyDeviceToHost);
    Eigen::Matrix<T, Eigen::Dynamic, 1> host_diagonal_with_lambda; host_diagonal_with_lambda.resize(nrows + nelem, 1);
    host_diagonal_with_lambda.topRows(nrows) = host_Hpp.diagonal();
    host_diagonal_with_lambda.bottomRows(nelem) = host_Hmm_diag;

    T scale = 0.5 * delta_x.transpose() * (host_diagonal_with_lambda.asDiagonal() * delta_x + b) + 1e-6;

    // T tempChi = 0.5 * (host_imu_chi2.sum() + host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum() + host_err_prior.squaredNorm());
    T tempChi = 0.5 * (host_imu_chi2.sum() + host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum());
    // T tempChi_1 = 0.5 * (host_imu_chi2.sum() + host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum());
    // T tempChi_2 = 0.5 * (host_proj_2f1c_chi2.sum() + host_proj_2f2c_chi2.sum() + host_proj_1f2c_chi2.sum());
    // T tempChi_3 = 0.5 * host_imu_chi2.sum();

    T rho = (currentChi - tempChi) / scale;
    if (rho > 0 && std::isfinite(tempChi))   // last step was good, error is dropping
    {
        T alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, T(2. / 3.));
        T scaleFactor = std::max(T(1. / 3.), alpha);
        currentLambda *= scaleFactor;
        cudaMemcpy(dev_ptr_lambda, &currentLambda, 1 * sizeof(T), cudaMemcpyHostToDevice);
        // printf("---------------------------------------- \n");
        // printf("Branch#1 \n");
        // printf("rho : %.20f \n", rho);
        // printf("currentChi : %.20f \n", currentChi);
        // printf("tempChi   : %.20f \n", tempChi);
        // printf("tempChi_1 : %.20f \n", tempChi_1);
        // printf("tempChi_2 : %.20f \n", tempChi_2);
        // printf("tempChi_3 : %.20f \n", tempChi_3);
        // printf("scale : %.20f \n", scale);
        // printf("currentLambda : %.20f \n", currentLambda);
        // std::cout << "host_err_prior      : " << host_err_prior.squaredNorm() / weight << std::endl;
        // std::cout << "host_imu_chi2       : " << host_imu_chi2.sum() << std::endl;
        // std::cout << "host_proj_2f1c_chi2 : " << host_proj_2f1c_chi2.sum() << std::endl;
        // std::cout << "host_proj_2f2c_chi2 : " << host_proj_2f2c_chi2.sum() << std::endl;
        // std::cout << "host_proj_1f2c_chi2 : " << host_proj_1f2c_chi2.sum() << std::endl;
        // std::cout << "b.array().abs().maxCoeff() : " << b.array().abs().maxCoeff() << std::endl;
        // printf("---------------------------------------- \n");
        ni = 2;     // 2
        currentChi = tempChi;
        num_consecutive_bad_steps = 0;
        return true;
    } else {
        currentLambda *= ni;
        cudaMemcpy(dev_ptr_lambda, &currentLambda, 1 * sizeof(T), cudaMemcpyHostToDevice);
        // printf("---------------------------------------- \n");
        // printf("Branch#2 \n");
        // printf("rho : %.20f \n", rho);
        // printf("currentChi : %.20f \n", currentChi);
        // printf("tempChi   : %.20f \n", tempChi);
        // printf("tempChi_1 : %.20f \n", tempChi_1);
        // printf("tempChi_2 : %.20f \n", tempChi_2);
        // printf("tempChi_3 : %.20f \n", tempChi_3);
        // printf("scale : %.20f \n", scale);
        // printf("currentLambda : %.20f \n", currentLambda);
        // std::cout << "host_err_prior      : " << host_err_prior.squaredNorm() / weight << std::endl;
        // std::cout << "host_imu_chi2       : " << host_imu_chi2.sum() << std::endl;
        // std::cout << "host_proj_2f1c_chi2 : " << host_proj_2f1c_chi2.sum() << std::endl;
        // std::cout << "host_proj_2f2c_chi2 : " << host_proj_2f2c_chi2.sum() << std::endl;
        // std::cout << "host_proj_1f2c_chi2 : " << host_proj_1f2c_chi2.sum() << std::endl;
        // std::cout << "b.array().abs().maxCoeff() : " << b.array().abs().maxCoeff() << std::endl;
        // printf("---------------------------------------- \n");
        ni *= 2;    // 2
        num_consecutive_bad_steps++;
        return false;
    }
}

// ------------------------------------------------------------------------------------------------------------------------

// instantiation
template struct LevenbergLambda<double>;
template struct LevenbergLambda<float>;

} // namespace VINS_FUSION_CUDA_BA
