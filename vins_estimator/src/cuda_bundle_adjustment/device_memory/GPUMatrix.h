//
// Created by lmf on 23-7-22.
//

#ifndef CUDA_BA_GPUMATRIX_H
#define CUDA_BA_GPUMATRIX_H

#include <cstdio>
#include <cassert>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace VINS_FUSION_CUDA_BA {

template<typename T> class GPUMemBlock2D;
template<typename T> class CUDAMatrix2D;
template<typename T, unsigned int NRows, unsigned int NCols> class CUDAMatrix2DStatic;

template<typename T>
class GPUMemBlock2D {

public :
    T* dev_ptr;
    int num_rows;
    int num_cols;
    int leading_dim;

public :
    GPUMemBlock2D();
    ~GPUMemBlock2D();
    bool Alloc(int nrows, int ncols, int ld, const std::string& err_desc = "ALLOC ERROR!");
    int rows() const;
    int cols() const;
    int ld() const;
    bool empty() const;

public :
    T* GetDevRawPtr();
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
class CUDAMatrix2D{
    typedef GPUMemBlock2D<T> GPUMemBlock2DType;

public :
    CUDAMatrix2D() = default;
    CUDAMatrix2D<T>& operator = (const CUDAMatrix2D<T>& other) = delete;
    int rows() const;
    int cols() const;
    int ld() const;
    bool empty() const;
    void Reset();
    bool Reset(int nrows, int ncols, int ld = 0, const std::string& err_desc = "ALLOC ERROR!");

public :
    T* GetDevRawPtr();

private :
    std::unique_ptr<GPUMemBlock2DType> dev_ptr;
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T, unsigned int NRows, unsigned int NCols>
class CUDAMatrix2DStatic{
    typedef GPUMemBlock2D<T> GPUMemBlock2DType;

public :
    CUDAMatrix2DStatic();
    int rows() const;
    int cols() const;
    int ld() const;

public :
    T* GetDevRawPtr();

public :
    std::unique_ptr<GPUMemBlock2DType> dev_ptr;
};

} // namespace VINS_FUSION_CUDA_BA

#include "GPUMatrix_impl.hpp"

#endif //CUDA_BA_GPUMATRIX_H
