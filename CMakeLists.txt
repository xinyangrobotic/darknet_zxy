cmake_minimum_required(VERSION 2.8)
project(dark_zxy)

set(CMAKE_CXX_STANDARD 11)
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

#OpenCV
find_package(OpenCV 3.4.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/libs/darknet)

add_executable(dark_zxy main.cpp)

target_link_libraries(dark_zxy
        ${OpenCV_LIBS}
        darknet )
