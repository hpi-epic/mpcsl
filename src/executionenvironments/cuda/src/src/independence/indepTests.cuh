#pragma once
#include "../util/indepUtil.h"
#include <vector>

__global__ void testL0(State state);

__global__ void testL1Naive(int i, int j, State state);

__host__ __device__ double calcPValue(double r, int sampleSize);

__host__ __device__ double pValL1(const double * cor, int i, int j,
                                  int k, int size, int sampleSize);

__global__ void testL1(State state, double* adj_out);
