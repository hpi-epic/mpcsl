#ifndef PARALLELPC_WORKER_H
#define PARALLELPC_WORKER_H


#include <memory>
#include <vector>

#include "concurrency.h"
#include "skeleton.h"



class Worker {
public:
    Worker(
        TaskQueue t_queue,
        std::shared_ptr<PCAlgorithm> alg,
        int level,
        std::shared_ptr<Graph> graph,
        std::shared_ptr<Graph> working_graph,
        std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> sep_matrix,
        std::shared_ptr<Statistics> statistics
    );

    // Task to fetch test from _work_queue
    // and put the results the working graph and the separation matrix.
    void execute_test();

    // Write independence test results to the graph and separation set store 
    inline void update_result(int x, int y, const std::vector<int> &subset) {
        increment_stat(_statistics->deleted_edges)
        _working_graph->deleteEdge(x, y);
        (*_separation_matrix)[x * _alg->getNumberOfVariables() + y] =  std::make_shared<std::vector<int> >(subset);
    }
    void test_single_conditional();
    void test_higher_order();

protected:
    TaskQueue _work_queue;
    std::shared_ptr<PCAlgorithm> _alg;
    int _level;
    std::shared_ptr<Graph> _graph;
    std::shared_ptr<Graph> _working_graph;
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<int>>>> _separation_matrix;
    std::shared_ptr<Statistics> _statistics;
};

#endif //PARALLELPC_WORKER_H