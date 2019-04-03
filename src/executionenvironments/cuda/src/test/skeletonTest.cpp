#include <gtest/gtest.h>
#include "testHelper.h"
#include "../src/correlation/corOwn.cuh"
#include "../src/independence/skeleton.cuh"

TEST(SkeletonTest, L0AndL1Separation) {
    auto inputVector = read_csv("data/coolingData.csv");

    int p = static_cast<int>(inputVector.size()),
        observations = static_cast<int>(inputVector[0].size()),
        maxCondSize = 1;
    double* mat = createArray(inputVector);

    double* cor = new double[p * p];
    gpuPMCCShared(mat, p, observations, cor);

    double* adj = new double[p * p];
    std::fill_n(adj, p * p, 1.0);

    SepSets calcSeparation = calcSkeleton(adj, cor, observations, p, 0.05,
        maxCondSize, 1);
    cudaDeviceSynchronize();

    SepSets actualSeparation;
    actualSeparation[1] = std::unordered_set<int> ();
    actualSeparation[2] = std::unordered_set<int> ();
    actualSeparation[29] = std::unordered_set<int> ({3});
    SepSets::iterator it = actualSeparation.begin();

    ASSERT_EQ(calcSeparation.size(), 3)
    << "Expected number of separations to be 3";
    while (it != actualSeparation.end()) {
        div_t divresult = div(it->first, p);
        ASSERT_EQ(calcSeparation[it->first], actualSeparation[it->first])
        << "Separation sets differ at edge "
        << it->first << " (Edge between " << divresult.quot
        << ", " << divresult.rem << ")";
        it++;
    }
}

TEST(SkeletonTest, NoSeparation) {
    auto inputVector = read_csv("data/5v_1-3-2.csv");

    int p = static_cast<int>(inputVector.size()),
        observations = static_cast<int>(inputVector[0].size()),
        maxCondSize = 4;
    double* mat = createArray(inputVector);

    double* cor = new double[p * p];
    gpuPMCCShared(mat, p, observations, cor);

    double* adj = new double[p * p];
    std::fill_n(adj, p * p, 1.0);

    SepSets calcSeparation = calcSkeleton(adj, cor, observations, p, 0.05,
        maxCondSize, 1);
    cudaDeviceSynchronize();

    ASSERT_EQ(calcSeparation.size(), 8)
    << "Expected map to contain 8 separations but has "
    << calcSeparation.size()
    << " instead!";

    SepSets::iterator it = calcSeparation.begin();

    while (it != calcSeparation.end()) {
        div_t divresult = div(it->first, p);
        ASSERT_EQ(it->second, std::unordered_set<int> ({}))
        << "Expected " << it->first << " (Edge between "
        << divresult.quot << ", " << divresult.rem
        << ") to be emtpy!";
        it++;
    }
}

TEST(SkeletonTest, AllLevels) {
    auto inputVector = read_csv("data/coolingData.csv");

    int p = static_cast<int>(inputVector.size()),
        observations = static_cast<int>(inputVector[0].size()),
        maxCondSize = 2;
    double* mat = createArray(inputVector);

    double* cor = new double[p * p];
    gpuPMCCShared(mat, p, observations, cor);

    double* adj = new double[p * p];
    std::fill_n(adj, p * p, 1.0);

    SepSets calcSeparation = calcSkeleton(adj, cor, observations, p, 0.05,
        maxCondSize, 1);
    cudaDeviceSynchronize();

    SepSets actualSeparation;
    // L0 and L1 Seps
    actualSeparation[1] = std::unordered_set<int> ();
    actualSeparation[2] = std::unordered_set<int> ();
    actualSeparation[29] = std::unordered_set<int> ({3});
    // L2
    actualSeparation[3] = std::unordered_set<int> ({4, 5});
    actualSeparation[4] = std::unordered_set<int> ({1, 3});
    actualSeparation[5] = std::unordered_set<int> ({1, 3});
    actualSeparation[10] = std::unordered_set<int> ({2, 3});
    actualSeparation[11] = std::unordered_set<int> ({2, 3});
    actualSeparation[16] = std::unordered_set<int> ({1, 3});
    actualSeparation[17] = std::unordered_set<int> ({1, 3});
    SepSets::iterator it = actualSeparation.begin();

    ASSERT_EQ(calcSeparation.size(), actualSeparation.size())
    << "Expected number of separations to be 10";
    while (it != actualSeparation.end()) {
        div_t divresult = div(it->first, p);
        ASSERT_EQ(calcSeparation[it->first], actualSeparation[it->first])
        << "Separation sets differ at edge "
        << it->first << " (Edge between " << divresult.quot
        << ", " << divresult.rem << ")";
        it++;
    }
}
