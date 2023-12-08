//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_CUDA_ERROR_CHECK_H
#define CUDA_BA_CUDA_ERROR_CHECK_H

#include <iostream>

#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(call, error_str)                                       \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if(error_code != cudaSuccess) {                                             \
        printf("CUDA Error : \n");                                              \
        printf("\tFile       : %s \n ", __FILE__);                              \
        printf("\tLine       : %d \n ", __LINE__);                              \
        printf("\tError Code : %d \n ", error_code);                            \
        printf("\tError Info : %s \n ", cudaGetErrorString(error_code));        \
        assert(error_code == cudaSuccess);                                      \
    }                                                                           \
} while(0)

#define PRINT_CUDA_ERROR(cuda_error_code)                                   \
do {                                                                        \
    printf("CUDA Error : \n");                                              \
    printf("\tFile       : %s \n ", __FILE__);                              \
    printf("\tLine       : %d \n ", __LINE__);                              \
    printf("\tError Code : %d \n ", cuda_error_code);                       \
    printf("\tError Info : %s \n ", cudaGetErrorString(cuda_error_code));   \
} while(0)

#define RETURN_FALSE_IF_CUDA_ERROR(call, description)                       \
do {                                                                        \
    const cudaError_t cuda_status = call;                                   \
    if(cuda_status != cudaSuccess) {                                        \
        std::cout << description << std::endl;                              \
        printf("CUDA Error : \n");                                          \
        printf("\tFile       : %s \n", __FILE__);                           \
        printf("\tLine       : %d \n", __LINE__);                           \
        printf("\tError Code : %d \n", cuda_status);                        \
        printf("\tError Info : %s \n", cudaGetErrorString(cuda_status));    \
        return false;                                                       \
    }                                                                       \
} while(0)

#define RETURN_ZERO_IF_CUDA_ERROR(call, description)                        \
do {                                                                        \
    const cudaError_t cuda_status = call;                                   \
    if(cuda_status != cudaSuccess) {                                        \
        printf(description.c_str()); printf(" \n");                         \
        printf("CUDA Error : \n");                                          \
        printf("\tFile       : %s \n", __FILE__);                           \
        printf("\tLine       : %d \n", __LINE__);                           \
        printf("\tError Code : %d \n", cuda_status);                        \
        printf("\tError Info : %s \n", cudaGetErrorString(cuda_status));    \
        return 0;                                                           \
    }                                                                       \
} while(0)

#endif //CUDA_BA_CUDA_ERROR_CHECK_H
