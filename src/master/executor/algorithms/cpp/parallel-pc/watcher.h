#ifndef PARALLELPC_WATCHER_H
#define PARALLELPC_WATCHER_H


#include <memory>
#include <vector>

#include "concurrency.h"
#include "skeleton.h"



class Watcher {
public:
    Watcher(
        std::vector<int> *stats,
        int max
    );

    void watch();

    void set_max(int new_max);

protected:
    std::vector<int> *_stats;
    int _max;
};

#endif //PARALLELPC_WATCHER_H