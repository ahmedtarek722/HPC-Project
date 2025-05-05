cmake_minimum_required(VERSION 3.10)
project(Project)

# Find OpenCV in standard locations
find_package(OpenCV REQUIRED)

# Print some information about OpenCV for debugging
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Project main.cpp)
target_link_libraries(Project ${OpenCV_LIBS})
