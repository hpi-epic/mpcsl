#include <gtest/gtest.h>
#include "testHelper.h"
#include "../src/correlation/corOwn.cuh"

TEST(CorrelationTest, Correct) {
    auto inputVector = read_csv("data/5v_1-3-2.csv");

    int p = static_cast<int>(inputVector.size()),
        observations = static_cast<int>(inputVector[0].size());
    double* mat = createArray(inputVector);

    double* cor = new double[p * p];
    gpuPMCC(mat, p, observations, cor);

    double actualCor[25] =
        {1.00000000, 0.05771775, 0.62195493, -0.06766349, -0.01669977,
        0.05771775, 1.00000000, 0.43355428, -0.02539817, 0.12212159,
        0.62195493, 0.43355428, 1.00000000, -0.05314783, 0.05161356,
        -0.06766349, -0.02539817, -0.05314783, 1.00000000, 0.09471016,
        -0.01669977, 0.12212159, 0.05161356, 0.09471016, 1.00000000};
    cudaDeviceSynchronize();

    for (int i = 0; i < p*p; ++i) {
      ASSERT_NEAR(cor[i],
        actualCor[i], 0.01)
        << "Calculated and predicted differ at index " << i;
    }
}
