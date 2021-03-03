cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nightsight)

# Download LibTorch for target platform from https://pytorch.org
find_package(Torch REQUIRED)

add_executable(nightsight inference.cpp)
target_link_libraries(nightsight ${TORCH_LIBRARIES})
set_property(TARGET nightsight PROPERTY CXX_STANDARD 14)

