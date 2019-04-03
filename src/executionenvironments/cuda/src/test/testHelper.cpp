#include "testHelper.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>

std::vector <std::vector<double>> read_csv(std::string filename) {
    std::ifstream csv(filename);
    std::vector <std::vector<double>> items;

    if (csv.is_open()) {
        bool init = false;
        for (std::string row_line; getline(csv, row_line);) {
            std::istringstream row_stream(row_line);
            int i = 0;
            for (std::string column; getline(row_stream, column, ','); i++) {
                if (!init)
                    items.push_back(std::vector<double>());
                items[i].push_back(static_cast<double> (stod(column)));
            }
            init = true;
        }
    } else {
        std::cout << "Unable to open file" << std::endl;
    }
    return items;
}

double* createArray(const std::vector <std::vector<double>>& v ) {
    double* result = new double[v.size() * v[0].size()];

    double* tempRes = result;
    for (int i =0; i < v.size(); i++) {
      std::copy(v[i].begin(), v[i].end(), tempRes);
      tempRes += v[i].size();
    }

    return result;
}

void printSeps(SepSets sep, int p) {
    SepSets::iterator it = sep.begin();
    while (it != sep.end()) {
        div_t divresult = div(it->first, p);
        std::cout
        << it->first << " (Edge between "
        << divresult.quot << ", " << divresult.rem
        << ") = ";
        for (auto i : it->second) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        it++;
    }
}
