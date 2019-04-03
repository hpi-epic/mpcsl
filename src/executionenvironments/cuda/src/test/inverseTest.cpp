#include <gtest/gtest.h>
#include "testHelper.h"
#include "../src/independence/pseudoInverse.cuh"

TEST(PseudoInverseTest, InverseForInts) {
    double mat[9] = { -1, 8, 2,
                    5, 6, -5,
                    -9, 0, 1};
    int kSize = 3;

    double* calcInv = new double[kSize * kSize];
    calculateMatrixPseudoInverse(mat, calcInv, kSize);

    double actualInv[9] = {0.0142, -0.0190, -0.1232,
                          0.0948, 0.0403, 0.0118,
                          0.1280, -0.1706, -0.1090};
    cudaDeviceSynchronize();

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(calcInv[i],
        actualInv[i], 0.001)
        << "Calculated and predicted differ at index " << i;
    }
    delete[] calcInv;
}

TEST(PseudoInverseTest, InverseForDoubles) {
    double mat[16] = { 1.000000, 0.851970, 0.948579, 0.859633,
                      0.851970, 1.000000, 0.836630, 0.992951,
                      0.948579, 0.836630, 1.000000, 0.844389,
                      0.859633, 0.992951, 0.844389, 1.000000};
    int kSize = 4;

    double* calcInv = new double[kSize * kSize];
    calculateMatrixPseudoInverse(mat, calcInv, kSize);

    double actualInv[16] = {11.3373, 0.1626, -8.7966, -2.4796,
                            0.1626, 71.2430, 0.3224, -71.1527,
                            -8.7966, 0.3224, 10.3123, -1.4658,
                            -2.4796, -71.1527, -1.4658, 75.0205};
    cudaDeviceSynchronize();

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(calcInv[i],
        actualInv[i], 0.001)
        << "Calculated and predicted differ at index " << i;
    }
    delete[] calcInv;
}

TEST(PseudoInverseTest, CorrectSVD) {
    double mat[16] = { 1.000000, 0.851970, 0.948579, 0.859633,
                      0.851970, 1.000000, 0.836630, 0.992951,
                      0.948579, 0.836630, 1.000000, 0.844389,
                      0.859633, 0.992951, 0.844389, 1.000000};
    int kSize = 4;

    double *d_U, *d_VT, *d_S,
        *h_U = new double[kSize * kSize],
        *h_VT = new double[kSize * kSize],
        *h_S = new double[kSize];

    cudaMalloc(reinterpret_cast<void **> (&d_U),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_VT),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_S),
               sizeof(double) * kSize);

    svd(mat, kSize, d_U, d_VT, d_S);
    cudaDeviceSynchronize();

    cudaMemcpy(h_U, d_U, sizeof(double) * kSize * kSize,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_VT, d_VT, sizeof(double) * kSize * kSize,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S, d_S, sizeof(double) * kSize,
               cudaMemcpyDeviceToHost);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_S);

    double actualVT[16] = {-0.4988, 0.4687, 0.7289, -0.0136,
                          -0.5022, -0.5105, -0.0285, -0.6974,
                          -0.4946, 0.5366, -0.6837, -0.0086,
                          -0.5043, -0.4815, -0.0221, 0.7165};

    double actualU[16] = {-0.4988, -0.5022, -0.4946, -0.5043,
                           0.4687, -0.5105, 0.5366, -0.4815,
                           0.7289, -0.0285, -0.6837, -0.0221,
                           -0.0136, -0.6974, -0.0086, 0.7165};

    double actualS[4] = {3.6673, 0.2749, 0.0509, 0.0069};

    for (int i = 0; i < kSize; ++i) {
      ASSERT_NEAR(h_S[i],
        actualS[i], 0.001)
        << "Calculated and predicted Vector S differ at index " << i;
    }

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(h_VT[i],
        actualVT[i], 0.001)
        << "Calculated and predicted Matrix VT differ at index " << i;
    }

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(h_U[i],
        actualU[i], 0.001)
        << "Calculated and predicted Matrix U differ at index " << i;
    }

    delete[] h_U;
    delete[] h_VT;
    delete[] h_S;
}

TEST(PseudoInverseTest, CorrectSInversion) {
    double h_S[4] = {3.6673, 0.2749, 0.0509, 0.0069};
    int kSize = 4;

    double *d_S, *d_S_inv,
      *h_S_inv = new double[kSize * kSize];

    cudaMalloc(reinterpret_cast<void **> (&d_S),
               sizeof(double) * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_S_inv),
               sizeof(double) * kSize * kSize);
    cudaMemcpy(d_S, h_S, sizeof(double) * kSize,
               cudaMemcpyHostToDevice);

    dim3 block(kSize), gridX(kSize);
    matrixInverse<<<gridX, block>>>(d_S, d_S_inv, kSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_S_inv, d_S_inv, sizeof(double) * kSize * kSize,
               cudaMemcpyDeviceToHost);
    cudaFree(d_S);
    cudaFree(d_S_inv);

    double actualS_inv[16] = {0.27268, 0, 0, 0,
                              0, 3.63769, 0, 0,
                              0, 0, 19.64637, 0,
                              0, 0, 0, 144.92754};

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(h_S_inv[i],
        actualS_inv[i], 0.001)
        << "Calculated and predicted Matrix U differ at index " << i;
    }
    delete[] h_S_inv;
}

TEST(PseudoInverseTest, CorrectMultiplication) {
    double h_VT[16] = {-0.4988, 0.4687, 0.7289, -0.0136,
                       -0.5022, -0.5105, -0.0285, -0.6974,
                       -0.4946, 0.5366, -0.6837, -0.0086,
                       -0.5043, -0.4815, -0.0221, 0.7165};

    double h_U[16] = {-0.4988, -0.5022, -0.4946, -0.5043,
                      0.4687, -0.5105, 0.5366, -0.4815,
                      0.7289, -0.0285, -0.6837, -0.0221,
                      -0.0136, -0.6974, -0.0086, 0.7165};

    double h_S_inv[16] = {0.27268, 0, 0, 0,
                          0, 3.63769, 0, 0,
                          0, 0, 19.64637, 0,
                          0, 0, 0, 144.92754};
    int kSize = 4;

    double *d_VT, *d_S_inv, *d_U,
      *h_pInv = new double[kSize * kSize];

    cudaMalloc(reinterpret_cast<void **> (&d_VT),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_U),
               sizeof(double) * kSize * kSize);
    cudaMalloc(reinterpret_cast<void **> (&d_S_inv),
               sizeof(double) * kSize * kSize);
    cudaMemcpy(d_U, h_U, sizeof(double) * kSize * kSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_VT, h_VT, sizeof(double) * kSize * kSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_inv, h_S_inv, sizeof(double) * kSize * kSize,
               cudaMemcpyHostToDevice);

    matrixMatrixMultiplication(d_VT, d_S_inv, d_U, kSize, h_pInv);
    cudaDeviceSynchronize();

    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_S_inv);

    double actualInv[16] = {11.3318, 0.1644, -8.7916, -2.481,
                            0.1644, 71.5207, 0.3224, -71.4428,
                            -8.7916, 0.3224, 10.3085, -1.468,
                            -2.481, -71.4428, -1.468, 75.3241};

    for (int i = 0; i < kSize*kSize; ++i) {
      ASSERT_NEAR(h_pInv[i],
        actualInv[i], 0.001)
        << "Calculated and predicted pseudoinverse matrix differ at index "
        << i;
    }
}
