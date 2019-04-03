#include <gtest/gtest.h>
#include "../src/util/indepUtil.h"

bool VERBOSE;

int main(int argc, char **argv) {
    VERBOSE = true;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
