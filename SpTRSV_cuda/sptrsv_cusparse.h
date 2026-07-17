#ifndef _SPTRSV_CUSPARSE_
#define _SPTRSV_CUSPARSE_

//#include "common.h"
//#include "utils.h"
//#include <cuda_runtime.h>


#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                              \
{                                                                     \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        printf("CUDA error %s:%d : %s\n",                             \
               __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

#define CHECK_CUSPARSE(call)                                          \
{                                                                     \
    cusparseStatus_t status = (call);                                 \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        printf("cuSPARSE error %s:%d\n", __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

void triangularSolve(
    int n,
    int nnz,
    const int *csrRowPtr,
    const int *csrColIdx,
    const double *csrVal,
    const double *rhs,
    double *solution,
    bool lower,
    bool unitDiagonal)
{
    //-----------------------------------------------------------------
    // cuSPARSE handle
    //-----------------------------------------------------------------

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    //-----------------------------------------------------------------
    // Copy data to GPU
    //-----------------------------------------------------------------

    int *dRowPtr, *dColIdx;
    double *dVal, *dX, *dB;

    CHECK_CUDA(cudaMalloc(&dRowPtr, (n+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dColIdx, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dVal, nnz*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dX, n*sizeof(double)));

    CHECK_CUDA(cudaMemcpy(dRowPtr, csrRowPtr,
                          (n+1)*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dColIdx, csrColIdx,
                          nnz*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dVal, csrVal,
                          nnz*sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dB, rhs,
                          n*sizeof(double), cudaMemcpyHostToDevice));

    //-----------------------------------------------------------------
    // Create sparse matrix descriptor
    //-----------------------------------------------------------------

    cusparseSpMatDescr_t matA;

    CHECK_CUSPARSE(
        cusparseCreateCsr(
            &matA,
            n,
            n,
            nnz,
            dRowPtr,
            dColIdx,
            dVal,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));

    cusparseFillMode_t fill =
        lower ? CUSPARSE_FILL_MODE_LOWER
              : CUSPARSE_FILL_MODE_UPPER;

    cusparseDiagType_t diag =
        unitDiagonal ? CUSPARSE_DIAG_TYPE_UNIT
                     : CUSPARSE_DIAG_TYPE_NON_UNIT;

    CHECK_CUSPARSE(
        cusparseSpMatSetAttribute(
            matA,
            CUSPARSE_SPMAT_FILL_MODE,
            &fill,
            sizeof(fill)));

    CHECK_CUSPARSE(
        cusparseSpMatSetAttribute(
            matA,
            CUSPARSE_SPMAT_DIAG_TYPE,
            &diag,
            sizeof(diag)));

    //-----------------------------------------------------------------
    // Dense vectors
    //-----------------------------------------------------------------

    cusparseDnMatDescr_t matB, matX;

    CHECK_CUSPARSE(
          cusparseCreateDnMat(
             &matB,
             n,                  // rows
             1,                  // columns
             n,                  // leading dimension
             dB,
             CUDA_R_64F,
             CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(
          cusparseCreateDnMat(
             &matX,
             n,
             1,
             n,
             dX,
             CUDA_R_64F,
             CUSPARSE_ORDER_COL));
    //-----------------------------------------------------------------
    // SpSM descriptor
    //-----------------------------------------------------------------

    cusparseSpSMDescr_t spsmDescr;
    CHECK_CUSPARSE(cusparseSpSM_createDescr(&spsmDescr));

    double alpha = 1.0;

    size_t bufferSize;

    CHECK_CUSPARSE(
        cusparseSpSM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            matX,
            CUDA_R_64F,
            CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr,
            &bufferSize));

    void *buffer;
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    //-----------------------------------------------------------------
    // Analysis
    //-----------------------------------------------------------------

    cudaDeviceSynchronize();

    printf(" - cusparse SpTRSV analysis start!\n");
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    CHECK_CUSPARSE(
        cusparseSpSM_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            matX,
            CUDA_R_64F,
            CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr,
            buffer));

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("cusparse SpTRSV analysis used %4.2f ms\n", time_cuda_analysis);
    //-----------------------------------------------------------------
    // Solve
    //-----------------------------------------------------------------

    printf(" - cusparse SpTRSV solve start!\n");
    gettimeofday(&t1, NULL);

    CHECK_CUSPARSE(
        cusparseSpSM_solve(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            matX,
            CUDA_R_64F,
            CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr));

    CHECK_CUDA(cudaMemcpy(solution,
                          dX,
                          n*sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_cuda_solve = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_solve /= BENCH_REPEAT;

    printf("cusparse SpTRSV solve used %4.2f ms\n", time_cuda_solve);

    //-----------------------------------------------------------------
    // Cleanup
    //-----------------------------------------------------------------

    cudaFree(buffer);

    cusparseSpSM_destroyDescr(spsmDescr);

    cusparseDestroyDnMat(matX);
    cusparseDestroyDnMat(matB);

    cusparseDestroySpMat(matA);

    cudaFree(dRowPtr);
    cudaFree(dColIdx);
    cudaFree(dVal);
    cudaFree(dX);
    cudaFree(dB);

    cusparseDestroy(handle);
}


#endif



