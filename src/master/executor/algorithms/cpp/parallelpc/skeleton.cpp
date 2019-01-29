#include "skeleton.h"
#include "worker.h"

PCAlgorithm::PCAlgorithm(int vars, double alpha, int samples, int numberThreads): _graph(std::make_shared<Graph>(vars)), _alpha(alpha), _nr_variables(vars), _nr_samples(samples), _nr_threads(numberThreads){
    _correlation = arma::Mat<double>(vars, vars, arma::fill::eye);
    _gauss_test = IndepTestGauss(_nr_samples,_correlation);
    _work_queue = std::make_shared<moodycamel::ConcurrentQueue<TestInstruction> >();
    _separation_matrix = std::make_shared<std::vector<std::shared_ptr<std::vector<int>>>>(_nr_variables*_nr_variables, nullptr);
}

void PCAlgorithm::build_graph() {

    long total_tests = 0;
    int level = 1;
    std::unordered_set<int> nodes_to_be_tested;
    for (int i = 0; i < _nr_variables; ++i) nodes_to_be_tested.insert(nodes_to_be_tested.end(), i);
    std::vector<int> stats(_nr_threads, 0);

    std::chrono::time_point<std::chrono::high_resolution_clock> start_queue, end_queue, start_worker, end_worker;
    
    cout << "Starting to fill test_queue" << endl;

    // we want to run as long as their are edges remaining to test on a higher level
    while(!nodes_to_be_tested.empty()) {
        set_time(start_queue)
        int queue_size = 0;
        std::vector<int> nodes_to_delete(0);
        // iterate over all edges to determine if they still can be tested on this level
        for (int x : nodes_to_be_tested) {
            if(_graph->getNeighbourCount(x)-1 >= level) {
                auto adj = _graph->getNeighbours(x);
                for (int &y : adj) {
                    if(y < x || _graph->getNeighbourCount(y)-1 < level) {
                        _work_queue->enqueue(TestInstruction{x, y});
                        queue_size++;
                    }
                }
            } else {
                // if they have not enough neighbors for this level, they won't on the following
                nodes_to_delete.push_back(x);
            }
            // only do the independence testing if the current_node has enough neighbours do create a separation set
        }
        set_time(end_queue)
        double duration_queue = 0.0;
        add_time_to(duration_queue, start_queue, end_queue)
        if(queue_size) {
            cout << "Queued all " << queue_size << " pairs, waiting for results.." << endl;

            vector<shared_ptr<thread> > threads;
            // we could think of making this a member variable and create the workers once and only the threads if they are needed
            vector<shared_ptr<Worker> > workers;
            vector<shared_ptr<Statistics> > stats(_nr_threads);

            set_time(start_worker)
            rep(i,_nr_threads) {
                stats[i] = std::make_shared<Statistics>();
                workers.push_back(make_shared<Worker>(
                    _work_queue,
                    shared_from_this(),
                    level,
                    _graph,
                    _working_graph,
                    _separation_matrix,
                    stats[i]
                ));
                threads.push_back(make_shared<thread>(&Worker::execute_test, *workers[i]));
            }

            for(const auto &thread : threads) {
                thread->join();
            }
            set_time(end_worker)
            double duration_worker = 0.0;
            add_time_to(duration_worker, start_worker, end_worker)
#ifdef WITH_STATS
            cout << "Duration queue fuelling: " << duration_queue << " s" << endl;
            cout << "Duration queue processing: " << duration_worker << " s" << endl;
            double tests_total = 0.0;
            double elements_total = 0.0;
            for(int i = 0; i < _nr_threads; i++) {
                // std::cout << "Thread " << i << ": " << stats[i]->dequed_elements << " dequed elements, "
                //           << stats[i]->deleted_edges << " deleted edges and " << stats[i]->test_count << " tests." << std::endl;
                // std::cout << "Thread " << i << ": " << stats[i]->sum_time_gaus*1000 << " ms for all tests and "
                //           << stats[i]->sum_time_queue_element*1000 << " ms for all queued elements in total" << std::endl;
                // std::cout << "Thread " << i << ": " << stats[i]->sum_time_gaus*1000/stats[i]->test_count << " ms per test on average and "
                //           << stats[i]->sum_time_queue_element*1000/stats[i]->dequed_elements << " ms per queue element on average" << std::endl;
                total_tests += stats[i]->test_count;
                tests_total += stats[i]->sum_time_gaus;
                elements_total += stats[i]->sum_time_queue_element;
            }

            cout << "Total time for tests " << tests_total << "s and total time for all workers: " << elements_total << "s." << endl;
            cout << "Percentage tests: " << (tests_total/elements_total)*100.0 << "%." << endl;

#endif
            cout << "All tests done for level " << level << '.' << endl;
            stats.resize(0);
        } else {
            cout << "No tests left for level " << level << '.' << endl;
            _graph = std::make_shared<Graph>(*_working_graph);
            break;
        }
        
        
        for(const auto node: nodes_to_delete) {
            nodes_to_be_tested.erase(node);
        }
        _graph = std::make_shared<Graph>(*_working_graph);
        level++;
    }

    cout << "Total independence tests made: " << total_tests << std::endl;
}

void PCAlgorithm::print_graph() const {
    _graph->print_list();
}

int PCAlgorithm::getNumberOfVariables() {
    return _nr_variables;
}

void PCAlgorithm::build_correlation_matrix(std::vector<std::vector<double>> &data) {
    int deleted_edges = 0;
    int n = data[0].size();
    rep(i, _nr_variables) {
        rep(j, i) {
            gsl_vector_const_view gsl_x = gsl_vector_const_view_array( &data[i][0], n);
            gsl_vector_const_view gsl_y = gsl_vector_const_view_array( &data[j][0], n);
            double pearson = gsl_stats_correlation(
                    gsl_x.vector.data, STRIDE,
                    gsl_y.vector.data, STRIDE,
                    n
            );
            _correlation(i,j) = _correlation(j,i) = pearson;
        }
    }
    _gauss_test = IndepTestGauss(_nr_samples,_correlation);

    std::vector<int> empty_sep(0);
    rep(i, _nr_variables) {
        rep(j, i) {
            auto pearson = _gauss_test.test(i, j, empty_sep);
            if(pearson >= _alpha) {
                deleted_edges += 2;
                _graph->deleteEdge(i,j);
            }
        }
    }
    cout << "Deleted edges: " << deleted_edges << std::endl;
    _working_graph = std::make_shared<Graph>(*_graph);
}

void PCAlgorithm::persist_result(
    const std::string data_name,
    const std::vector<std::string> &column_names
) {
    std::function<std::string(int)> _node = [&column_names] (int i) {return std::to_string(i);};
    if (column_names.size() == _correlation.n_rows)
        _node = [&column_names] (int i) {return column_names[i];};

    // Create dir
    std::string filename = data_name;
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
        filename.erase(0, last_slash_idx + 1);

    // Remove extension if present.
    const size_t period_idx = filename.rfind('.');
    if (std::string::npos != period_idx)
        filename.erase(period_idx);

    std::string dir_name = "results_" + filename + "_" + std::to_string(_alpha) + "/";
    system(("mkdir -p " + dir_name).c_str());

    // Save nodes
    ofstream node_file;
    node_file.open(dir_name + "nodes.txt");
    for (const auto c : column_names) {
        node_file << c << std::endl;
    }

    // Save graph
    ofstream graph_file;
    graph_file.open(dir_name + "skeleton.txt");

    for (int i = 0; i < _correlation.n_rows; i++) {
        for (const auto j : _graph->getNeighbours(i)) {
            graph_file << _node(i) << " " << _node(j) << std::endl;
        }
    }

    // Save correlation matrix
    _correlation.save(dir_name + "corr.csv" , arma::csv_ascii);

    // Save separation set information
    ofstream sepset_file;
    sepset_file.open(dir_name + "sepset.txt");

    rep(i, _nr_variables) {
        rep(j, _nr_variables) {
            auto pt = (*_separation_matrix)[i * _nr_variables + j];
            if (pt != nullptr) {
                sepset_file << i << " " << j << " ";
                for (auto const s : *pt) {
                    sepset_file << s << ' ';
                }
                sepset_file << std::endl;
            }
        }
    }
}
