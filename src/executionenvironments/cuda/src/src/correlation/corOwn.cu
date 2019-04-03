#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "corOwn.cuh"
#include "corHelper.cuh"
#include "../util/indepUtil.h"

#define NUMTHREADS 32
#define PERTHREAD 8

void gpuPMCC(const double * h_mat, int n,
    int dim, double * h_cors) {
  if (VERBOSE)
    printf("Cor started with N=%i, dim=%i\n", n, dim);
  size_t
    dbytes = sizeof(double);
  double
    *d_mat, *d_means, *d_stddevs,
    *d_cors;
  dim3
    block(NUMTHREADS), grid(n, n), gridX(n);

  cudaMalloc(reinterpret_cast<void **>(&d_means), n * dbytes);
  cudaMalloc(reinterpret_cast<void **>(&d_stddevs), n * dbytes);
  cudaMalloc(reinterpret_cast<void **>(&d_cors), n*n*dbytes);

  cudaMalloc(reinterpret_cast<void **>(&d_mat), n*dim*dbytes);
  cudaMemcpy(d_mat, h_mat, n*dim*dbytes, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  gpuMeans<<<gridX, block>>>(d_mat, n, dim, d_means);
  cudaThreadSynchronize();
  if (VERBOSE)
    printf("Means successful \n");

  gpuSD<<<gridX, block>>>(d_mat, n, dim, d_means, d_stddevs);
  cudaThreadSynchronize();
  if (VERBOSE)
    printf("SD successful \n");

  gpuPMCC<<<grid, block>>>(d_mat, n, dim, d_means, d_stddevs, d_cors);
  cudaMemcpy(h_cors, d_cors, n*n*dbytes,
    cudaMemcpyDeviceToHost);
  if (VERBOSE)
    printf("PMCC successful \n");

  // Free allocated space
  cudaFree(d_means);
  cudaFree(d_stddevs);
  cudaFree(d_cors);
  cudaFree(d_mat);
}


void gpuPMCCShared(const double * h_mat, int n,
    int dim, double * h_cors) {
  size_t
    dbytes = sizeof(double);
  double
    *d_mat, *d_means, *d_stddevs,
    *d_cors;
  size_t gridY = ((n%PERTHREAD == 0) ? n/PERTHREAD : (n/PERTHREAD) + 1);
  dim3
    block(NUMTHREADS), grid(n, gridY), gridX(n);
  cudaMalloc(reinterpret_cast<void **>(&d_means), n * dbytes);
  cudaMalloc(reinterpret_cast<void **>(&d_stddevs), n * dbytes);
  cudaMalloc(reinterpret_cast<void **>(&d_cors), n*n*dbytes);

  cudaMalloc(reinterpret_cast<void **>(&d_mat), n*dim*dbytes);
  cudaMemcpy(d_mat, h_mat, n*dim*dbytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  gpuMeans<<<gridX, block>>>(d_mat, n, dim, d_means);
  cudaThreadSynchronize();

  gpuSD<<<gridX, block>>>(d_mat, n, dim, d_means, d_stddevs);
  cudaThreadSynchronize();

  gpuPMCCShared<<<grid, block>>>(d_mat, n, dim, d_means, d_stddevs, d_cors);
  cudaMemcpy(h_cors, d_cors, n*n*dbytes,
    cudaMemcpyDeviceToHost);

  // Free allocated space
  cudaFree(d_means);
  cudaFree(d_stddevs);
  cudaFree(d_cors);
  cudaFree(d_mat);
}
