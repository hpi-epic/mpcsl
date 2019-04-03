#include <vector>
#include <string>
#include "../src/util/indepUtil.h"

std::vector <std::vector<double>> read_csv(std::string filename);

double* createArray(const std::vector <std::vector<double>>& v);

void printSeps(SepSets sep, int p);
