//
// Created by lmf on 23-7-20.
//

#ifndef CUDA_BA_COMMON_H
#define CUDA_BA_COMMON_H

namespace VINS_FUSION_CUDA_BA {

enum StateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

} // namespace VINS_FUSION_CUDA_BA

#endif //CUDA_BA_COMMON_H
