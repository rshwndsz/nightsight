cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Nightsight
    VERSION 0.1.0 
    DESCRIPTION "Low light enhancement of images"
    HOMEPAGE_URL "https://github.com/rshwndsz/nightsight"
    LANGUAGES CXX
)

# Specify the C++ standard
# TODO Just specify the exact features used and let cmake decide standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Download LibTorch for target platform from https://pytorch.org
find_package(Torch 
    REQUIRED 
    PATHS
        ${PROJECT_SOURCE_DIR}/include/libtorch
)

# Pass version number to source code
configure_file(include/NightsightConfig.h.in include/NightsightConfig.h)

add_executable(nightsight 
    src/main.cpp)

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(nightsight
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(nightsight 
    # No "default behaviour" - https://stackoverflow.com/a/51404894
        ${TORCH_LIBRARIES}
)

