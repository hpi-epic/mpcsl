#ifndef PARALLELPC_SKELETON_H
#define PARALLELPC_SKELETON_H


#define rep(a, b)   for(int a = 0; a < (b); ++a)

#include "graph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>

#include <armadillo>

#include "concurrency.h"
#include "condition.h"


using namespace std;

class PCAlgorithm : public enable_shared_from_this<PCAlgorithm> {

public:
    PCAlgorithm(int vars, double alpha, int samples, int numberThreads);
    double _alpha;
    void build_graph();

    void print_graph() const;

    int getNumberOfVariables();

    void build_correlation_matrix(std::vector<std::vector<double>> &data);

    inline double test(int u, int v, std::vector<int> &S) const {
        return _gauss_test.test(u,v,S);
    }

    void persist_result(const std::string data_name, const std::vector<std::string> &column_names);


protected:

    TaskQueue _work_queue;
    IndepTestGauss _gauss_test;
    const int STRIDE = 1;
    int _nr_variables;
    int _nr_samples;
    int _nr_threads;
    arma::Mat<double>_correlation;
    shared_ptr<Graph> _graph;
    shared_ptr<Graph> _working_graph;
    shared_ptr<vector<shared_ptr<vector<int>>>> _separation_matrix;

    void print_vector(const vector<int> &S) const {
        for(auto s : S)
            cout << s << " ";
        cout << endl;
    }
};

#endif //PARALLELPC_SKELETON_H
