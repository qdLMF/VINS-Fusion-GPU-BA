//
// Created by lmf on 23-8-24.
//

#include "utility.h"

namespace VINS_FUSION_CUDA_BA {

Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3d ypr2R(const Eigen::Vector3d& ypr)
{
    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0.0,
          sin(y),  cos(y), 0.0,
             0.0,     0.0, 1.0;

    Eigen::Matrix3d Ry;
    Ry <<  cos(p), 0.0, sin(p),
              0.0, 1.0,    0.0,
          -sin(p), 0.0, cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1.0,    0.0,     0.0,
          0.0, cos(r), -sin(r),
          0.0, sin(r),  cos(r);

    return Rz * Ry * Rx;
}


Eigen::Vector3f R2ypr(const Eigen::Matrix3f& R)
{
    Eigen::Vector3f n = R.col(0);
    Eigen::Vector3f o = R.col(1);
    Eigen::Vector3f a = R.col(2);

    Eigen::Vector3f ypr(3);
    float y = atan2(n(1), n(0));
    float p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    float r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3f ypr2R(const Eigen::Vector3f& ypr)
{
    float y = ypr(0) / 180.0 * M_PI;
    float p = ypr(1) / 180.0 * M_PI;
    float r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix3f Rz;
    Rz << cos(y), -sin(y), 0.0,
          sin(y),  cos(y), 0.0,
             0.0,     0.0, 1.0;

    Eigen::Matrix3f Ry;
    Ry <<  cos(p), 0.0, sin(p),
              0.0, 1.0,    0.0,
          -sin(p), 0.0, cos(p);

    Eigen::Matrix3f Rx;
    Rx << 1.0,    0.0,     0.0,
          0.0, cos(r), -sin(r),
          0.0, sin(r),  cos(r);

    return Rz * Ry * Rx;
}

} // namespace VINS_FUSION_CUDA_BA

