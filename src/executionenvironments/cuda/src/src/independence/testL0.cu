#include "testL0.cuh"
#include "indepTests.cuh"
#include <chrono>

TestResult indTestL0(State h_state) {
    double* d_adj, *d_cor, *d_pMax;
    int* d_sepSets;
    cudaMalloc(reinterpret_cast<void **> (&d_adj),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_cor),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_pMax),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_sepSets),
               sizeof(int) * h_state.p * h_state.p);
    cudaMemcpy(d_adj, h_state.adj, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_cor, h_state.cor, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sepSets, h_state.sepSets,
               sizeof(int) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    State d_state = { d_pMax, d_adj, d_cor, d_sepSets, h_state.p,
                      h_state.observations, h_state.alpha,
                      h_state.maxCondSize };
    int numthreads = min(d_state.p, 32);
    dim3 block(numthreads), grid(d_state.p);
    auto start = std::chrono::system_clock::now();

    testL0<<<grid, block>>>(d_state);
    cudaDeviceSynchronize();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    cudaMemcpy(h_state.pMax, d_state.pMax,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.adj, d_state.adj,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.sepSets, d_state.sepSets,
               sizeof(int) * h_state.p * h_state.p,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_sepSets);
    cudaFree(d_adj);
    cudaFree(d_cor);
    cudaFree(d_pMax);
    return { static_cast<unsigned long>(duration),
             (h_state.p * (h_state.p - 1)) / 2 };
}
