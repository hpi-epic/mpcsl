#include "../util/indepUtil.h"

SepSets calcSkeleton(double * adjMatrix, double * cor,
                  int observations, int p, double alpha,
                  int maxCondSize, int NAdelete);

void updateMap(SepSets *sepSets, State state);
