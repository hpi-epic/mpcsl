#include "pseudoInverse.cuh"
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

void calculateMatrixPseudoInverse(double *tmpMat, double *inverse, int kSize) {
    double *d_U, *d_VT, *d_S, *d_S_inv;

    cudaMalloc(reinterpret_cast<void **> (&d_U),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_VT),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_S),
               sizeof(double) * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_S_inv),
               sizeof(double) * kSize * kSize);

    svd(tmpMat, kSize, d_U, d_VT, d_S);

    dim3 block(kSize), gridX(kSize);
    matrixInverse<<<gridX, block>>>(d_S, d_S_inv, kSize);
    cudaDeviceSynchronize();

    matrixMatrixMultiplication(d_VT, d_S_inv, d_U, kSize, inverse);

    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_S);
    cudaFree(d_S_inv);
}

void svd(double *tmpMat, int kSize, double *d_U, double *d_VT, double *d_S) {
    cusolverDnHandle_t cusolverH;
    int lwork = 0;
    double *d_work, *d_rwork, *d_A;
    int *devInfo = NULL;
    char jobu = 'A';
    char jobvt = 'A';


    cudaMalloc(reinterpret_cast<void **> (&d_A),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&devInfo), sizeof(int));
    cudaMemcpy(d_A, tmpMat, sizeof(double) * kSize * kSize,
               cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    cusolverDnDgesvd_bufferSize(cusolverH, kSize, kSize, &lwork);

    cudaMalloc(reinterpret_cast<void **> (&d_work), sizeof(double)*lwork);
    cudaMalloc(reinterpret_cast<void **> (&d_rwork), sizeof(double)*lwork);
    cusolverDnDgesvd(cusolverH, jobu, jobvt, kSize, kSize,
        d_A, kSize, d_S, d_U, kSize, d_VT,
        kSize, d_work, lwork, d_rwork, devInfo);

    cudaFree(d_work);
    cudaFree(d_rwork);
    cudaFree(d_A);
    cudaFree(devInfo);
}

void matrixMatrixMultiplication(double *d_VT, double * d_S_inv,
                                double *d_U, int kSize, double *inverse) {
    const double factor = 1.0;
    const double beta = 0.0;
    double *tmpRes, *result;

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cudaMalloc(reinterpret_cast<void **> (&tmpRes),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&result),
               sizeof(double) * kSize * kSize);

    // cublas matmul transpose(VT) * s_inv = tmpRes
    cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, kSize, kSize, kSize, &factor,
        d_VT, kSize, d_S_inv, kSize, &beta, tmpRes, kSize);

    // cublas matmul tmpRes * transpose(U) = tmpMat
    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, kSize, kSize, kSize, &factor,
        tmpRes, kSize, d_U, kSize, &beta, result, kSize);
    cudaDeviceSynchronize();

    cudaMemcpy(inverse, result, sizeof(double)*kSize * kSize,
               cudaMemcpyDeviceToHost);
    cudaFree(tmpRes);
    cudaFree(result);
}

__global__ void matrixInverse(double *S, double *S_inv, int kSize) {
    int bx = blockIdx.x, tx = threadIdx.x;

    if (bx == tx) {
        S_inv[bx * kSize + tx] = 1.f / S[bx];
    } else {
        S_inv[bx * kSize + tx] = 0.f;
    }
}
