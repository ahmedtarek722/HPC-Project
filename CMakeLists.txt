cmake_minimum_required(VERSION 3.10)
project(Project LANGUAGES CXX)

# -- Find OpenCV
find_package(OpenCV REQUIRED)

# -- Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# -- Include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${MPI_CXX_INCLUDE_DIRS}
)

# -- Executable
add_executable(Project mpi.cpp)

# -- Link libraries
target_link_libraries(Project PRIVATE
  ${OpenCV_LIBS}
  ${MPI_CXX_LIBRARIES}
)
