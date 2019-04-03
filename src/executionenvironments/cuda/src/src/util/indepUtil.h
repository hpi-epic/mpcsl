#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>

typedef std::unordered_map<int, std::unordered_set<int>> SepSets;

template <typename Iterator>
bool next_combination(const Iterator first, Iterator k, const Iterator last) {
    // Credits: Mark Nelson http://marknelson.us
    if ((first == last) || (first == k) || (last == k))
        return false;
    Iterator i1 = first;
    Iterator i2 = last;
    ++i1;
    if (last == i1)
        return false;
    i1 = last;
    --i1;
    i1 = k;
    --i2;
    while (first != i1) {
        if (*--i1 < *i2) {
            Iterator j = k;
            while (!(*i1 < *j)) ++j;
            std::iter_swap(i1, j);
            ++i1;
            ++j;
            i2 = k;
            std::rotate(i1, j, last);
            while (last != j) {
                ++j;
                ++i2;
            }
            std::rotate(k, i2, last);
            return true;
        }
    }
    std::rotate(first, k, last);
    return false;
}

/**
Saves every important value/structure needed for the indepTests.

@param pMax Partial correlation values. Gets updated with every additional level of the tests.
@param adj Adjacency matrix. It is used to reduce the problem, in case we have knowledge of the structure.
@param cor Correlation matrix.
@param sepSet Separation sets to determine which nodes are separating two others. (Data structure maybe change.)
@param observations Number of observations.
@param p number of variables.

*/
struct State {
    double* pMax;
    double* adj;
    const double* cor;
    int* sepSets;
    int p;
    int observations;
    double alpha;
    int maxCondSize;
};

struct TestResult {
    unsigned long duration;
    int edges;
};

extern bool VERBOSE;
