#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cuda_fp16.h>

#include <stdio.h>

void* dBuffer = NULL;
cusparseHandle_t     handle = NULL;

void create_buffer(int A_num_rows, int A_num_cols, int A_nnz, int *dA_csrOffsets, int *dA_columns, __half *dA_values,
             int B_num_cols, int ldb, int ldc, __half *dB, __half *dC, float alpha, float beta) {
    // spmm: dA, dB, dC must be on GPUs
    // ldb, ldc: leading dimension of Matrix B and C
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    size_t               bufferSize = 0;

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                    CUDA_R_16F, CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_CSR_ALG3, &bufferSize);

    // printf("BufferSize: %ld\n", bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
}

void spmm_csr(int A_num_rows, int A_num_cols, int A_nnz, int *dA_csrOffsets, int *dA_columns, __half *dA_values,
             int B_num_cols, int ldb, int ldc, __half *dB, __half *dC, float alpha, float beta) {
    // spmm: dA, dB, dC must be on GPUs
    // ldb, ldc: leading dimension of Matrix B and C
    // CUSPARSE APIs
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                    CUDA_R_16F, CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL);

    // preprocess speedup
    // cusparseSpMM_preprocess(handle,
    //                         CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         &alpha, matA, matB, &beta, matC, CUDA_R_32F,
    //                         CUSPARSE_SPMM_CSR_ALG1, dBuffer);

    // execute SpMM
    cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG3, dBuffer) ;

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
}

void delete_cuSparse_Buffer() {
  if (dBuffer != NULL)
    cudaFree(dBuffer);
    cusparseDestroy(handle);
}

void create_cuSparse_Buffer(torch::Tensor x, torch::Tensor W, torch::Tensor res) {
  if (dBuffer == NULL)
  {
    cusparseCreate(&handle);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // get PyTorch tensor data pointer
    // convert c10::Half to CUDA __half
    auto x_data_half = x.data_ptr<at::Half>();
    auto x_data_cuda_half = reinterpret_cast<__half*>(x_data_half);

    auto res_data_half = res.data_ptr<at::Half>();
    auto res_data_cuda_half = reinterpret_cast<__half*>(res_data_half);

    auto W_data_half = W.values().data_ptr<at::Half>();
    auto W_data_cuda_half = reinterpret_cast<__half*>(W_data_half);

    create_buffer(
      W.size(0), W.size(1), W._nnz(), W.crow_indices().data_ptr<int>(), W.col_indices().data_ptr<int>(), W_data_cuda_half, 
      x.size(0), x.size(1), res.size(1), x_data_cuda_half, res_data_cuda_half, 1.0f, 0.0f
    );
  }
}


// A_num_rows = shape[0], cols = shape[1]
void csrSpmm_cuSparse(
  torch::Tensor x, torch::Tensor W, torch::Tensor res
) {
// x*W^T -> res
// W is A, x is B, res is C
// Device Protect
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

  // get PyTorch tensor data pointer
  // convert c10::Half to CUDA __half
  auto x_data_half = x.data_ptr<at::Half>();
  auto x_data_cuda_half = reinterpret_cast<__half*>(x_data_half);

  auto res_data_half = res.data_ptr<at::Half>();
  auto res_data_cuda_half = reinterpret_cast<__half*>(res_data_half);

  auto W_data_half = W.values().data_ptr<at::Half>();
  auto W_data_cuda_half = reinterpret_cast<__half*>(W_data_half);

  spmm_csr(
    W.size(0), W.size(1), W._nnz(), W.crow_indices().data_ptr<int>(), W.col_indices().data_ptr<int>(), W_data_cuda_half, 
    x.size(0), x.size(1), res.size(1), x_data_cuda_half, res_data_cuda_half, 1.0f, 0.0f
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csrSpmm_cuSparse", &csrSpmm_cuSparse, "Csr Spmm (cuSparse)");
  m.def("create_cuSparse_Buffer", &create_cuSparse_Buffer, "Create cuSparse Buffer");
  m.def("delete_cuSparse_Buffer", &delete_cuSparse_Buffer, "Delete cuSparse Buffer");
}
