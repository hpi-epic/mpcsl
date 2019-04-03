#include <vector>
#include "../util/indepUtil.h"

TestResult indTestGen(State state, int lvl, SepSets *seps);

bool checkNeighbours(std::vector<int> neighbours, int lvl,
                     int i, int j, State state, SepSets *seps);

double pMaxGen(int i, int j, int *neighbours, int kSize, State state);

double pValGenTest(double * mat, int kSize, int sampleSize);
