#include "skeleton.cuh"
#include <iostream>
#include <unordered_set>
#include "testL0.cuh"
#include "testL1.cuh"
#include "testGen.cuh"
#include "../util/matrixPrint.cuh"

SepSets calcSkeleton(double * adjMatrix, double * cor,
            int observations, int p, double alpha,
            int maxCondSize, int NAdelete) {
    double* pMax = new double[p * p];
    int* sepSetArray = new int[p * p];
    std::fill_n(sepSetArray, p * p, 0);

    struct State state = {pMax, adjMatrix, cor, sepSetArray, p,
                          observations, alpha, maxCondSize};
    SepSets sepSets;

    if (VERBOSE)
      std::cout << "maxCondSize: " << state.maxCondSize <<
          "  observations: " << state.observations <<
          "  p: " << state.p << "  NAdelete: " << NAdelete << std::endl;

    TestResult res = indTestL0(state);
    for (int i = 0; i < state.p; i++) {
        for (int j = i + 1; j < state.p; j++) {
            if (!state.adj[i * p + j]) {
                sepSets[i * p + j] = std::unordered_set<int>();
                if (VERBOSE) {
                    std::cout << "Separation from " << i << " to " << j
                    << " on L0" << std::endl;
                }
            }
        }
    }

    if (VERBOSE)
      std::cout << "Order 0 finished on " << res.edges <<
          " edges in " << res.duration << " microseconds." << std::endl;

    res = indTestL1(state);

    if (VERBOSE)
      std::cout << "Order 1 finished on " << res.edges <<
          " edges in " << res.duration << " microseconds." << std::endl;
    cudaDeviceSynchronize();

    updateMap(&sepSets, state);

    for (int lvl = 2; lvl <= state.maxCondSize; ++lvl) {
        res = indTestGen(state, lvl, &sepSets);
        if (VERBOSE)
          std::cout << "Order " << lvl << " finished on " << res.edges <<
               " edges in " << res.duration << " microseconds." << std::endl;
    }
    return sepSets;
}

void updateMap(SepSets *sepSets, State state) {
    int nrSeps = 0;
    for (int i = 0; i < state.p; i++) {
        for (int j = i + 1; j < state.p; j++) {
            std::unordered_set<int> seps;
            int sepSetElement = state.sepSets[i * state.p + j];
            if (sepSetElement) {
                seps.insert(sepSetElement);
            }
            if (VERBOSE) {
                if (!seps.empty()) {
                    std::cout << "Separation from " << i
                              << " to " << j << " via";
                    for (auto k : seps) {
                        std::cout << " " << k;
                    }
                    std::cout << std::endl;
                    ++nrSeps;
                }
            }
            if (!seps.empty()) {
                (*sepSets)[i * state.p + j] = seps;
            }
        }
    }
    if (VERBOSE)
      std::cout << "Total Number of Edges: " << nrSeps << std::endl;
}

