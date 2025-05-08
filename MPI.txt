cmake_minimum_required(VERSION 3.10)
project(Project LANGUAGES CXX)

# -- Find OpenCV
find_package(OpenCV REQUIRED)

# -- Find MPI (we’ll still need the headers/libs)
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

# (somewhere after find_package(MPI REQUIRED) …)

# --- hard-code or cache the exact path
set(MPIEXEC_PATH
    "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe"
    CACHE FILEPATH "Full path to mpiexec.exe on Windows"
)

# --- tell CMake to look there
find_program(MPIEXEC
    NAMES mpiexec.exe
    PATHS "${MPIEXEC_PATH}"
    NO_DEFAULT_PATH
)

# … later, in your run target …
add_custom_target(run
  COMMAND "${MPIEXEC}" -n ${NP} $<TARGET_FILE:Project>
  DEPENDS Project
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Project with ${NP} MPI processes"
)
