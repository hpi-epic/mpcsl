#include <cuda_runtime_api.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "indepTests.cuh"

__global__ void testL0(State state) {
    int bx = blockIdx.x, tx = threadIdx.x, bd = gridDim.x;
    if (bx < tx) {
        if (state.adj[bx * bd + tx]) {
            double pVal = calcPValue(state.cor[bx * bd + tx],
                                                  state.observations);
            state.pMax[bx * bd + tx] = pVal;
            state.pMax[tx * bd + bx] = pVal;
            if (state.pMax[bx * bd + tx] >= state.alpha) {
                state.adj[bx * bd + tx] = 0.f;
                state.adj[tx * bd + bx] = 0.f;
            }
        }
    }
}

__global__ void testL1Naive(int i, int j, State state) {
    int bx = blockIdx.x, tx = threadIdx.x, bd = blockDim.x;
    if (bx < tx) {
        int neighboursOf = i;
        int edgeTo = j;
        if (bx) {
            neighboursOf = j;
            edgeTo = i;
        }
        if (tx != edgeTo && tx != neighboursOf &&
                state.pMax[neighboursOf * bd + tx] < state.alpha) {
            state.pMax[neighboursOf * bd + edgeTo] = pValL1(state.cor,
                                                            neighboursOf,
                                                            edgeTo, tx, bd,
                                                            state.observations);
            if (state.pMax[neighboursOf * bd + edgeTo] >= state.alpha) {
                state.sepSets[neighboursOf * bd + edgeTo] = tx;
                state.adj[neighboursOf * bd + edgeTo] = 0.f;
                state.adj[edgeTo * bd + neighboursOf] = 0.f;
            }
        }
    }
}

/**
Calculates the p value for a given sampleSize and partial correlation
*/

__host__ __device__ double calcPValue(double r, int sampleSize) {
    r = min(0.9999999f, abs(r));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2 * r / (1 - r));
    return 2 * (1 - normcdf(absz));
}

__host__ __device__ double pValL1(const double * cor, int i, int j,
                                  int k, int size, int sampleSize) {
    double r = (cor[i * size + j] - cor[i * size + k] * cor[j * size + k]) /
        sqrt((1 - cor[j * size + k] * cor[j * size + k]) *
             (1 - cor[i * size + k] * cor[i * size + k]));
    return calcPValue(r, sampleSize);
}



__global__ void testL1(State state, double* adj_out) {
    // We keep input and output adjacencies separate to keep order correct
    // bx and by describe features, whose relationship is suspect to be tested
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
    const int numThreads = blockDim.x;
    // only consider lower triangular
    if (bx < by) {
        extern __shared__ double pVals[];
        // only consider conditioning variables (cond) that are not the same as
        // current nodes to be tested, also cond should not exceed features
        for (int offset = tx; offset < state.p; offset += numThreads) {
            pVals[tx] = -1;
            if (bx != offset && by != offset) {
                // check if edge still exists and if edge
                // is available to separation set
                if (state.adj[bx * state.p + by] != 0 &&
                    (state.adj[bx * state.p + offset] != 0 ||
                     state.adj[by * state.p + offset] != 0) ) {
                    pVals[tx] = pValL1(state.cor, bx, by, offset,
                                       state.p, state.observations);
                }
            }
            __syncthreads();
            if (tx == 0) {
                for (int i = 0; i < numThreads; ++i) {
                    if (pVals[i] > state.pMax[bx * state.p + by]) {
                        state.pMax[bx * state.p + by] = pVals[i];
                        if (pVals[i] >= state.alpha) {
                            // CAREFUL CURRENTLY LIMIT sepsets to Size 1
                            // as we only go through level 0 and 1
                            state.sepSets[bx * state.p + by] = offset + i;
                            adj_out[bx * state.p + by] =
                            adj_out[by * state.p + bx] = 0.f;
                            break;
                        }
                    }
                }
                // can we leave the function early here ?
            }
            __syncthreads();
        }
    }
}
