cmake_minimum_required(VERSION 3.15)
project(seam_carving)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

set(OpenCV_DIR /installation/OpenCV-/lib/cmake/opencv4/)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(seam_carving src/seam_carving.cpp)
target_link_libraries(seam_carving ${OpenCV_LIBS})
