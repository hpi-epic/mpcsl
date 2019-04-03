#include "rinterface.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include "correlation/corOwn.cuh"
#include "independence/skeleton.cuh"
#include "util/indepUtil.h"

bool VERBOSE;

// [[Rcpp::export]]
SepSets estimateSkeleton(Rcpp::NumericMatrix mat,
        const float alpha = 0.05, const int maxCondSize = 4,
        const int NAdelete = 1, const bool verbose = false) {
    VERBOSE = verbose;
    int p = mat.ncol(),
        observations = mat.nrow();

    Rcpp::NumericMatrix cor = Rcpp::NumericMatrix(mat.ncol(), mat.ncol());
    gpuPMCC(mat.begin(), mat.ncol(), mat.nrow(), cor.begin());

    Rcpp::NumericMatrix adj = Rcpp::NumericMatrix(mat.ncol(), mat.ncol());
    std::fill(adj.begin(), adj.end(), 1);

    SepSets seps = calcSkeleton(adj.begin(), cor.begin(), observations, p,
        alpha, maxCondSize, NAdelete);

    return seps;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix rpmccOwn(const Rcpp::NumericMatrix mat) {
  Rcpp::NumericMatrix cor = Rcpp::NumericMatrix(mat.ncol(), mat.ncol());
  gpuPMCC(mat.begin(), mat.ncol(), mat.nrow(), cor.begin());
  return cor;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix rpmccOwnShared(const Rcpp::NumericMatrix mat) {
  Rcpp::NumericMatrix cor = Rcpp::NumericMatrix(mat.ncol(), mat.ncol());
  gpuPMCCShared(mat.begin(), mat.ncol(), mat.nrow(), cor.begin());
  return cor;
}
