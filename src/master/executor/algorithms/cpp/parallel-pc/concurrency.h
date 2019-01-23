#pragma once 

#include <vector>
#include <memory>
#include "concurrentqueue/concurrentqueue.h"

#if WITH_STATS
#define increment_stat(x) x += 1;
#define set_time(x) x = chrono::high_resolution_clock::now();
#define add_time_to(x, start, stop) x += chrono::duration_cast<chrono::duration<double>>(stop - start).count();
#else
#define increment_stat(x)
#define set_time(x)
#define add_time_to(x, start, stop)
#endif

struct TestInstruction {
    int X;
    int Y;
};

struct Statistics{
    int test_count = 0;
    int dequed_elements = 0;
    int deleted_edges = 0;
    double sum_time_gaus = 0.0;
    double sum_time_queue_element = 0.0;
};

using TaskQueue = std::shared_ptr<moodycamel::ConcurrentQueue<TestInstruction> >;
