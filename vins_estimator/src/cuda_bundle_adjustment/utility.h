//
// Created by lmf on 23-8-24.
//

#ifndef CUDA_BA_UTILITY_H
#define CUDA_BA_UTILITY_H

#include <eigen3/Eigen/Dense>

namespace VINS_FUSION_CUDA_BA {

Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R);
Eigen::Matrix3d ypr2R(const Eigen::Vector3d& ypr);

Eigen::Vector3f R2ypr(const Eigen::Matrix3f& R);
Eigen::Matrix3f ypr2R(const Eigen::Vector3f& ypr);

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_UTILITY_H
