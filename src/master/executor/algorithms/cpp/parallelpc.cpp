#include <iostream>
#include <vector>
#include <thread>

#include "mpci_utils.h"

#include "parallelpc/condition.h"
#include "parallelpc/worker.h"
#include "parallelpc/graph.hpp"
#include "parallelpc/concurrency.h"
#include "parallelpc/skeleton.h"

#include "parallelpc/concurrentqueue/blockingconcurrentqueue.h"

int main(int argc, char* argv[]) {
    const char *filename;
    int nr_threads;
    double alpha = 0.01;


    if (argc == 3 || argc == 4) {
        istringstream s1(argv[1]);
        if (!(s1 >> nr_threads))
            cerr << "Invalid number " << argv[1] << '\n';
       filename = argv[2];
       if (argc == 4) {
           istringstream s2(argv[3]);
           if (!(s2 >> alpha))
               cerr << "Invalid number " << argv[3] << '\n';
       }
    } else {
        cout << "Usage: ./ParallelPC.out <number_of_threads> <filename> [alpha=0.01]" << std::endl;
        return 1;
    }

    std::ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.precision(10);

    string _match(filename);
    std::vector<std::vector<double> > data;
    vector<std::string> column_names(0);
    if (_match.find(".csv") != std::string::npos) {
        data = read_csv(filename, column_names);
    } else if (_match.find(".data") != std::string::npos) {
        data = read_data(filename);
    } else {
        std::cout << "Cannot process file '" << filename << "\'." << std::endl;
        std::cout << "Has to be .csv or .data format." << std::endl;

        return 1;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_graph,start_correlation,start, end_graph,end_correlation,end;
    auto alg = make_shared<PCAlgorithm>(data.size(), alpha, data[0].size(), nr_threads);

    set_time(start);
    set_time(start_correlation);
    alg->build_correlation_matrix(data);
    set_time(end_correlation);

    set_time(start_graph)
    alg->build_graph();
    set_time(end_graph)
    set_time(end);

    alg->print_graph();
    double duration = 0.0;
    double duration_graph = 0.0;
    double duration_correlation = 0.0;
    add_time_to(duration, start, end)
    add_time_to(duration_correlation, start_correlation, end_correlation)
    add_time_to(duration_graph, start_graph, end_graph)
    std::cout << "Total time algo: " << duration << "s" << std::endl;
    std::cout << "Total time correlation: " << duration_correlation << "s" << std::endl;
    std::cout << "Total time graph: " << duration_graph << "s" << std::endl;

    alg->persist_result(filename, column_names);
    cout.flush();
    return 0;
}
