#include "testGen.cuh"
#include <chrono>
#include <iostream>
#include <unordered_set>
// #include <omp.h>
#include "pseudoInverse.cuh"
#include "indepTests.cuh"


TestResult indTestGen(State state, int lvl, SepSets *seps) {
    int c = 0;
    auto start = std::chrono::system_clock::now();
//  #pragma omp parallel for shared(state, lvl) reduction(+:c) default(none)
    for (int i = 0; i < state.p; ++i) {
        for (int j = i + 1; j < state.p; ++j) {
            if (state.adj[i * state.p + j] &&
                    state.pMax[i * state.p + j] < state.alpha) {
                ++c;
                // check all neighbors of i and j
                // add them to a list to get combinations
                std::vector<int> neighbours;
                for (int k = 0; k < state.p; ++k) {
                    if ((k != j && k != i) &&
                      ((state.pMax[i * state.p + k] < state.alpha &&
                        state.adj[i * state.p + k]) ||
                      (state.pMax[j * state.p + k] < state.alpha &&
                        state.adj[j * state.p + k])))
                            neighbours.push_back(k);
                }
                // we have enough candidates to build potential separation sets
                checkNeighbours(neighbours, lvl, i, j, state, seps);
            }
        }
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    return { static_cast<unsigned long>(duration), c };
}

bool checkNeighbours(std::vector<int> neighbours, int lvl,
                     int i, int j, State state, SepSets *seps) {
    if (neighbours.size() >= lvl) {
        do {
            state.pMax[i *state.p + j] = pMaxGen(i, j, &neighbours[0],
                                                 lvl, state);
            if (state.pMax[i * state.p + j] >= state.alpha) {
                std::unordered_set<int> separation(neighbours.begin(),
                                              neighbours.begin() + lvl);
                (*seps)[i * state.p + j] = separation;
                state.adj[i * state.p + j] = 0.f;
                state.adj[j * state.p + i] = 0.f;
                return true;
            }
        } while (next_combination(neighbours.begin(),
                 neighbours.begin() + lvl,
                 neighbours.end()));
    }
    return false;
}

double pMaxGen(int i, int j, int *neighbours, int kSize, State state) {
    // The total number of nodes and size of the submatrix
    int numNodes = kSize + 2;
    double * tmpMat = new double[numNodes * numNodes];
    // Save all relevant nodes (i, j and neighbours)
    // to get an index on them and fill the submatrix
    int * nodes = new int[numNodes];
    nodes[0] = i;
    nodes[1] = j;
    for (int n = 2; n < numNodes; n++) {
        nodes[n] = neighbours[n - 2];
    }

    for (int row = 0; row < numNodes; row++) {
        for (int col = 0; col < numNodes; col++) {
            tmpMat[row * numNodes + col] = state.cor[nodes[col] *
                                           state.p + nodes[row]];
        }
    }

    calculateMatrixPseudoInverse(tmpMat, tmpMat, numNodes);

    // CUDA http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd
    // https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse
    double pVal = pValGenTest(tmpMat, kSize, state.observations);
    if (VERBOSE) {
      std::cout << "pMax " << i << ", " << j << ": " << pVal << " (Sep";
      for (int i = 0; i < kSize; i++) {
        std::cout << " " << neighbours[i];
      }
      std::cout << ")" << std::endl;
    }
    delete[] tmpMat;
    delete[] nodes;
    return pVal;
}

double pValGenTest(double * mat, int kSize, int sampleSize) {
    double r = -mat[1] / sqrt(mat[0] * mat[kSize + 2 + 1]);
    return calcPValue(r, sampleSize);
}
