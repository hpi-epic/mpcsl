#include "corHelper.cuh"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


#define NUMTHREADS 32
// #define NUMTHREADS 128
#define PERTHREAD 8

__global__ void gpuMeans(const double * d_mat, size_t n,
    size_t dim, double * d_means) {
    size_t
        offset, stride,
        bx = blockIdx.x, tx = threadIdx.x;
    double a;

    __shared__ double
        threadSums[NUMTHREADS];
    __shared__ int
        count[NUMTHREADS];

    if (bx >= n)
        return;

    threadSums[tx] = 0.f;
    count[tx] = 0;

    for (offset = tx; offset < dim; offset += NUMTHREADS) {
        a = d_mat[bx * dim + offset];
        if (!isnan(a)) {
            threadSums[tx] += a;
            count[tx] += 1;
        }
    }
    __syncthreads();

    for (stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            threadSums[tx] += threadSums[tx + stride];
            count[tx] += count[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        d_means[bx] = threadSums[0] / count[0];
    }
}

__global__ void gpuSD(const double * d_mat, size_t n,
    size_t dim, const double * d_means, double * d_stddevs) {
    size_t
        offset, stride,
        tx = threadIdx.x,
        bx = blockIdx.x;
    double
        a, termA;
    __shared__ double
        meanA,
        threadSums[NUMTHREADS];

    if (bx >= n)
        return;

    if (tx == 0) {
        meanA = d_means[bx];
    }
    __syncthreads();

    threadSums[tx] = 0.f;
    for (offset = tx; offset < dim; offset += NUMTHREADS) {
        a = d_mat[bx * dim + offset];
        if (!isnan(a)) {
            termA = a - meanA;
            threadSums[tx] += termA * termA;
        }
    }
    __syncthreads();

    for (stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            threadSums[tx] += threadSums[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0) {
        d_stddevs[bx] = sqrtf(threadSums[0] / (dim - 1.f));
    }
}

__global__ void gpuPMCC(const double * d_mat, size_t n,
    size_t dim, const double * d_means, const double * d_stddevs,
    double * d_cors) {
    size_t
        offset, stride,
        bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x;
    double
        a, b, scoreA, scoreB;
    __shared__ double
        meanA, meanB,
        sdA, sdB,
        threadSums[NUMTHREADS];

    if ((bx >= n) || (by >= n))
        return;

    if (tx == 0) {
        meanA = d_means[bx];
        meanB = d_means[by];
        sdA = d_stddevs[bx];
        sdB = d_stddevs[by];
    }
    __syncthreads();

    threadSums[tx] = 0.f;
    for (offset = tx; offset < dim; offset += NUMTHREADS) {
        a = d_mat[bx * dim + offset];
        b = d_mat[by * dim + offset];
        if (!(isnan(a) || isnan(b))) {
            scoreA = (a - meanA) / sdA;
            scoreB = (b - meanB) / sdB;
            threadSums[tx] += scoreA * scoreB;
        }
    }
    __syncthreads();

    for (stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) threadSums[tx] += threadSums[tx + stride];
        __syncthreads();
    }
    if (tx == 0) d_cors[bx*n + by] = threadSums[0] / (dim - 1.f);
}

__global__ void gpuPMCCShared(const double * d_mat, size_t n,
    size_t dim, const double * d_means, const double * d_stddevs,
    double * d_cors) {
    size_t
        offset, stride,
        bx = blockIdx.x, by = (blockIdx.y * PERTHREAD),
        tx = threadIdx.x;
    double
        a, b, scoreA, scoreB;
    __shared__ double
        meanA, meanB[PERTHREAD],
        sdA, sdB[PERTHREAD],
        threadSums[PERTHREAD][NUMTHREADS];

    if ((bx >= n) || (by >= n))
        return;

    if (tx == 0) {
        meanA = d_means[bx];
        sdA = d_stddevs[bx];
        for (int i = 0; i < PERTHREAD; ++i) {
            if (by + i < n) {
                meanB[i] = d_means[by + i];
                sdB[i] = d_stddevs[by + i];
            }
        }
    }
    __syncthreads();
    for (int i = 0; i < PERTHREAD; ++i) {
        threadSums[i][tx] = 0.f;
    }
    for (offset = tx; offset < dim; offset += NUMTHREADS) {
        a = d_mat[bx * dim + offset];
        for (int i = 0; i < PERTHREAD; ++i) {
            if (by + i < n) {
                b = d_mat[(by+i) * dim + offset];
                if (!(isnan(a) || isnan(b))) {
                    scoreA = (a - meanA) / sdA;
                    scoreB = (b - meanB[i]) / sdB[i];
                    threadSums[i][tx] += scoreA * scoreB;
                }
            }
        }
    }
    __syncthreads();

    for (stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            for (int i = 0; i < PERTHREAD; ++i) {
                threadSums[i][tx] += threadSums[i][tx + stride];
            }
        }
        __syncthreads();
    }
    if (tx == 0) {
        for (int i = 0; i < PERTHREAD; ++i) {
            if (by + i < n) {
                d_cors[bx*n + by + i] = threadSums[i][0] / (dim - 1.f);
            }
        }
    }
}
