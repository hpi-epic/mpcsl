# ---- IMPORTANT -----
# In order for this to work, you may need to manually adjust the path to your nvcc
# in case FindCUDA does not properly find it.
#
# To do this, after you executed `cmake CMakeLists.txt` once, open up the CMakeCache.txt
# in the same folder and adjust the following lines:
#
# //CUDA compiler
# CMAKE_CUDA_COMPILER:FILEPATH=CMAKE_CUDA_COMPILER-NOTFOUND
#
# To make it look something like this:
#
# //CUDA compiler
# CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc

project(runTests)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Locate CUDA and NVRTC
find_package(CUDA QUIET REQUIRED)
enable_language(CUDA)
include_directories(${CUDA_TOOLKIT_ROOT_DIR})
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

#Locate R
find_package(LibR REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB_RECURSE SOURCES
    "../src/*.cu"
    "../src/*.cuh")

file(GLOB TEST_FILES
    "./*.cpp"
    "./*.h")

set_source_files_properties( skeletonTest.cpp inverseTest.cpp correlationTest.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
cuda_add_executable(runTests main.cpp ${TEST_FILES} ${SOURCES})
target_include_directories(runTests PUBLIC ${LIBR_INCLUDE_DIRS})
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NVRTC_LIB} ${LIBR_CORE_LIBRARY})
