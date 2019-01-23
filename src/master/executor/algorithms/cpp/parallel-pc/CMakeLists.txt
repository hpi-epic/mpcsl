cmake_minimum_required(VERSION 3.5)
project(ParallelPC)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_BUILD_TYPE Release)

option(WITH_STATS "Output stats" ON)

if(WITH_STATS)
    add_definitions(-DWITH_STATS)
endif()

set(SOURCE_FILES main.cpp condition.h condition.cpp graph.hpp graph.cpp worker.h worker.cpp skeleton.h skeleton.cpp)

message("${CMAKE_MODULES_PATH}")

add_executable(ParallelPC ${SOURCE_FILES})

set_target_properties(
    ParallelPC
    PROPERTIES 
        SUFFIX ".out"
)

find_package(GSL REQUIRED)

find_package(Armadillo REQUIRED)
find_package(Boost REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

target_link_libraries(ParallelPC
    pthread
    GSL::gsl
    GSL::gslcblas
    ${ARMADILLO_LIBRARIES}
)
