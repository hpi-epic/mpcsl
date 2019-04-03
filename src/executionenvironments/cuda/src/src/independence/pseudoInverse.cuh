void calculateMatrixPseudoInverse(double *tmpMat, double *inverse, int kSize);

void svd(double *tmpMat, int kSize, double *d_U, double *d_VT, double *d_S);

void matrixMatrixMultiplication(double *d_VT, double * d_S_inv,
                                double *d_U, int kSize, double *inverse);

__global__ void matrixInverse(double *S, double *S_inv, int kSize);
