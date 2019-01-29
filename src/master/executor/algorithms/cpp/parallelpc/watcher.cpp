#include <chrono>
#include <thread>

#include "watcher.h"

Watcher::Watcher(
    std::vector<int> *stats,
    int max
) :
    _stats(stats),
    _max(max)
{}

void Watcher::watch() {

    std::chrono::seconds sec(1);

    while(true) {
        if (_max) {
            int done = 0;
            for(auto const tests: (*_stats)) {
                done += tests;
            }
            std::cout << done << " out of " << _max << std::endl;
            std::cout << (double) _max / (double) done << " p. done." << std::endl;
            std::cout << "\r\r";
        }

        std::this_thread::sleep_for(sec);
    }
}

void Watcher::set_max(int new_max) {
    _max = new_max;
}
