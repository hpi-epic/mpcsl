#include <Rcpp.h>

void rpmccOwn(const Rcpp::NumericMatrix h_mat, const int n,
              const int dim, Rcpp::NumericMatrix h_cors);

void rpmccOwnShared(const Rcpp::NumericMatrix h_mat,
                    const int n, const int dim, Rcpp::NumericMatrix h_cors);

void estimateSkeleton(Rcpp::NumericMatrix adjMatrix, Rcpp::NumericMatrix cor,
                      Rcpp::NumericMatrix sepSets, const int observations,
                      const int p, const float alpha, const int maxCondSize,
                      const int NAdelete);
