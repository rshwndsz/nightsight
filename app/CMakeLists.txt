cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(Nightsight
    VERSION 0.1.0
    DESCRIPTION "Low light enhancement of images"
    HOMEPAGE_URL "https://github.com/rshwndsz/nightsight"
    LANGUAGES CXX
)

include(GNUInstallDirs)

# Download LibTorch for target platform from https://pytorch.org
find_package(Torch
    REQUIRED
    PATHS
        ${PROJECT_SOURCE_DIR}/include/external/libtorch
)

# Pass version number to source code
configure_file(config.h.in config.h)

add_executable(nightsight
    src/main.cpp)
target_compile_features(nightsight PRIVATE cxx_std_17)

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(nightsight
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/external
        ${PROJECT_BINARY_DIR}
)

target_link_libraries(nightsight
    # No "default behaviour" - https://stackoverflow.com/a/51404894
        ${TORCH_LIBRARIES}
)
