cmake_minimum_required(VERSION 3.10)
project(Project LANGUAGES CXX)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Your executable
add_executable(${PROJECT_NAME} openmp.cpp)

# Link against OpenCV and OpenMP
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)
