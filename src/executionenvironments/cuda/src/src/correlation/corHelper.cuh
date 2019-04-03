__global__ void gpuMeans(const double * d_mat, size_t n,
    size_t dim, double * d_means);

__global__ void gpuSD(const double * d_mat, size_t n,
    size_t dim, const double * d_means, double * d_stddevs);

__global__ void gpuPMCC(const double * d_mat, size_t n,
    size_t dim, const double * d_means, const double * d_stddevs,
    double * d_cors);

__global__ void gpuPMCCShared(const double * d_mat, size_t n,
    size_t dim, const double * d_means, const double * d_stddevs,
    double * d_cors);
