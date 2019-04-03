#include <iostream>

void printMatrix(int m, int n, const double*A, const char* name) {
  for (int row = 0 ; row < m ; row++) {
    for (int col = 0 ; col < n ; col++) {
      double Areg = A[row + col*m];
      printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
    }
  }
}

void printGPUMatrix(int m, int n, const double*d_A, const char* name) {
  double *h_A = new double[m*n];
  cudaMemcpy(h_A, d_A, sizeof(double)*m * n,
             cudaMemcpyDeviceToHost);
  for (int row = 0 ; row < m ; row++) {
    for (int col = 0 ; col < n ; col++) {
      double Areg = h_A[row + col*m];
      printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
    }
  }
}
