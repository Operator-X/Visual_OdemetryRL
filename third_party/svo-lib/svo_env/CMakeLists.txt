project(svo_env)
cmake_minimum_required(VERSION 2.8.3)

# Set PYTHONPATH by modifying the environment
set(ENV{PYTHONPATH} "/usr/bin/python3:$ENV{PYTHONPATH}")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d video)  # [rlvo] only needed modules (all-modules pulls brew glog -> flag clash)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenMP)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)  # [rlvo] pybind11 3.x needs Development.Module
find_package(pybind11 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

set(HEADERS
  include/svo_env/svo_env.h
  include/svo_env/svo_factory.h
  include/svo_env/visualizer.h
)

set(SOURCES
  src/svo_vec_env.cpp
  src/svo_env.cpp
  src/svo_factory.cpp
  src/visualizer.cpp
  src/utils.cpp
)

# Interface Library
add_library(svo_env_lib src/svo_vec_env.cpp src/svo_env.cpp src/svo_factory.cpp src/visualizer.cpp)
include_directories(include)
target_link_libraries(svo_env_lib ${OpenCV_LIBRARIES} svo_common svo gflags OpenMP::OpenMP_CXX pybind11::module)  # [rlvo] pybind11 headers not on a system path on macOS

# Test Executable
# add_executable(svo_test src/svo_test.cpp)
# target_link_libraries(svo_test svo_env_lib gflags Boost::filesystem)

# Pybind library
pybind11_add_module(svo_env MODULE src/pybind_wrapper.cpp ${SOURCES})
target_link_libraries(svo_env PUBLIC svo_env_lib)
