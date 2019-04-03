#include "testL1.cuh"
#include <chrono>
#include <iostream>
#include "indepTests.cuh"

TestResult indTestL1Naive(State h_state) {
    double* d_adj, *d_cor, *d_pMax;
    int* d_sepSets;
    auto start_cp_in = std::chrono::system_clock::now();
    cudaMalloc(reinterpret_cast<void **> (&d_adj),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_cor),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_pMax),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **> (&d_sepSets),
               sizeof(int) * h_state.p * h_state.p);
    cudaMemcpy(d_adj, h_state.adj,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cor, h_state.cor,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pMax, h_state.pMax,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sepSets, h_state.sepSets,
               sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto duration_cp_in = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start_cp_in).count();
    if (VERBOSE)
      std::cout << "Copy to device " << duration_cp_in << " microseconds." << std::endl;
    State d_state = { d_pMax, d_adj, d_cor, d_sepSets, h_state.p,
                      h_state.observations, h_state.alpha,
                      h_state.maxCondSize };
    int c = 0;
    int numthreads = min(d_state.p, 32);
    size_t gridY = ((d_state.p % numthreads == 0) ?
                    d_state.p/numthreads : (d_state.p/numthreads) + 1);
    dim3 block(numthreads), grid(2, gridY);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < h_state.p; ++i) {
        for (int j = i + 1; j < h_state.p; ++j) {
            if (h_state.adj[i * h_state.p + j] &&
                h_state.pMax[i * h_state.p + j] < h_state.alpha) {
                ++c;
                testL1Naive<<<grid, block>>>(i, j, d_state);
            }
        }
    }
    cudaDeviceSynchronize();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();

    auto start_cp_out = std::chrono::system_clock::now();
    cudaMemcpy(h_state.pMax, d_state.pMax,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.adj, d_state.adj,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.sepSets, d_state.sepSets,
               sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyDeviceToHost);
    cudaFree(d_adj);
    cudaFree(d_cor);
    cudaFree(d_pMax);
    cudaFree(d_sepSets);
    cudaDeviceSynchronize();
    auto duration_cp_out = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start_cp_out).count();
    if (VERBOSE)
      std::cout << "Copy to device " << duration_cp_out << " microseconds." << std::endl;
    return { static_cast<unsigned long>(duration), c };
}

TestResult indTestL1(State h_state) {
    double* d_adj, *d_cor, *d_pMax, *d_adj_out;
    int* d_sepSets;
    auto start_cp_in = std::chrono::system_clock::now();
    cudaMalloc(reinterpret_cast<void **>(&d_adj),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **>(&d_adj_out),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **>(&d_cor),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **>(&d_pMax),
               sizeof(double) * h_state.p * h_state.p);
    cudaMalloc(reinterpret_cast<void **>(&d_sepSets),
               sizeof(int) * h_state.p * h_state.p);
    cudaMemcpy(d_adj, h_state.adj, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_out, d_adj, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cor, h_state.cor, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_pMax, h_state.pMax, sizeof(double) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sepSets, h_state.sepSets,
               sizeof(int) * h_state.p * h_state.p,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto duration_cp_in = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start_cp_in).count();
    if (VERBOSE)
      std::cout << "Copy to device in microseconds: " << duration_cp_in << std::endl;
    State d_state = { d_pMax, d_adj, d_cor, d_sepSets, h_state.p,
                      h_state.observations, h_state.alpha,
                      h_state.maxCondSize };
    int c = 0;
    // TODO(Siegfried): correct c
    int numthreads = min(d_state.p, 32);
    dim3 block(numthreads), grid(h_state.p, h_state.p);
    auto start = std::chrono::system_clock::now();
    testL1<<<grid, block, sizeof(double)*numthreads>>>(d_state, d_adj_out);
    cudaDeviceSynchronize();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    auto start_cp_out = std::chrono::system_clock::now();
    cudaMemcpy(h_state.pMax, d_state.pMax,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.adj, d_adj_out,
               sizeof(double) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state.sepSets, d_state.sepSets,
               sizeof(int) * h_state.p * h_state.p, cudaMemcpyDeviceToHost);
    cudaFree(d_adj);
    cudaFree(d_adj_out);
    cudaFree(d_cor);
    cudaFree(d_pMax);
    cudaFree(d_sepSets);
    cudaDeviceSynchronize();
    auto duration_cp_out = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start_cp_out).count();
    if (VERBOSE)
      std::cout << "Copy from device in microseconds: " << duration_cp_out << std::endl;
    return { static_cast<unsigned long>(duration), c};
}
