//
// Created by lmf on 23-7-22.
//

#include <cstdio>
#include <cassert>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "GPUMatrix.h"
#include "cuda_error_check.h"

namespace VINS_FUSION_CUDA_BA {

template<typename T>
GPUMemBlock2D<T>::GPUMemBlock2D() : dev_ptr(nullptr), num_rows(0), num_cols(0), leading_dim(0) {}

template<typename T>
bool GPUMemBlock2D<T>::Alloc(int nrows, int ncols, int ld, const std::string& err_desc) {
    ld = ld == 0 ? nrows : ld;

    assert(nrows > 0);
    assert(ncols > 0);
    assert(nrows <= ld);

    num_rows = nrows;
    num_cols = ncols;
    leading_dim = ld;

//    cudaError_t cuda_status;

    dev_ptr = nullptr;
    RETURN_FALSE_IF_CUDA_ERROR(cudaMalloc((void**)&dev_ptr, sizeof(T) * leading_dim * num_cols), err_desc);
    RETURN_FALSE_IF_CUDA_ERROR(cudaMemset(dev_ptr, 0, sizeof(T) * leading_dim * num_cols), err_desc);

    return true;
}

template<typename T>
GPUMemBlock2D<T>::~GPUMemBlock2D() {
    if(dev_ptr) {
        cudaFree(dev_ptr);
        dev_ptr = nullptr;
    }
}

template<typename T>
int GPUMemBlock2D<T>::rows() const {
    return num_rows;
}

template<typename T>
int GPUMemBlock2D<T>::cols() const {
    return num_cols;
}

template<typename T>
int GPUMemBlock2D<T>::ld() const {
    return leading_dim;
}

template<typename T>
bool GPUMemBlock2D<T>::empty() const {
    return (dev_ptr == nullptr) || (num_rows == 0) || (num_cols == 0) || (leading_dim == 0);
}

template<typename T>
T* GPUMemBlock2D<T>::GetDevRawPtr() {
    return dev_ptr;
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T>
int CUDAMatrix2D<T>::rows() const {
    return dev_ptr->rows();
}

template<typename T>
int CUDAMatrix2D<T>::cols() const {
    return dev_ptr->cols();
}

template<typename T>
int CUDAMatrix2D<T>::ld() const {
    return dev_ptr->ld();
}

template<typename T>
bool CUDAMatrix2D<T>::empty() const {
    return dev_ptr->empty();
}

template<typename T>
T* CUDAMatrix2D<T>::GetDevRawPtr() {
    return dev_ptr->GetDevRawPtr();
};

template<typename T>
void CUDAMatrix2D<T>::Reset() {
    dev_ptr.reset();
};

template<typename T>
bool CUDAMatrix2D<T>::Reset(int nrows, int ncols, int ld, const std::string& err_desc) {
    ld = ld == 0 ? nrows : ld;

    assert(nrows > 0);
    assert(ncols > 0);
    assert(nrows <= ld);

    dev_ptr.reset();
    dev_ptr = std::make_unique<GPUMemBlock2DType>();
    bool success = dev_ptr->Alloc(nrows, ncols, ld, err_desc);

    return success;
};

// ------------------------------------------------------------------------------------------------------------------------

template<typename T, unsigned int NRows, unsigned int NCols>
CUDAMatrix2DStatic<T, NRows, NCols>::CUDAMatrix2DStatic() {
    dev_ptr = std::make_unique<GPUMemBlock2DType>();
    dev_ptr->Alloc(NRows, NCols, NRows);
}

template<typename T, unsigned int NRows, unsigned int NCols>
int CUDAMatrix2DStatic<T, NRows, NCols>::rows() const {
    return dev_ptr->rows();
}

template<typename T, unsigned int NRows, unsigned int NCols>
int CUDAMatrix2DStatic<T, NRows, NCols>::cols() const {
    return dev_ptr->cols();
}

template<typename T, unsigned int NRows, unsigned int NCols>
int CUDAMatrix2DStatic<T, NRows, NCols>::ld() const {
    return dev_ptr->ld();
}

template<typename T, unsigned int NRows, unsigned int NCols>
T *CUDAMatrix2DStatic<T, NRows, NCols>::GetDevRawPtr() {
    return dev_ptr->GetDevRawPtr();
}

} // namespace VINS_FUSION_CUDA_BA

