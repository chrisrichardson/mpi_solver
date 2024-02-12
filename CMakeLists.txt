cmake_minimum_required(VERSION 3.18)

project(mpi_solver)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cpp
  NOMINSIZE
  wrapper.cpp
  superlu.cpp
)

find_package(MPI REQUIRED)
target_include_directories(cpp PRIVATE ${MPI_C_INCLUDE_DIRS})
target_include_directories(cpp PRIVATE /Users/chrisrichardson/Work/packages/petsc/include)
target_link_directories(cpp PRIVATE /Users/chrisrichardson/Work/packages/petsc/lib)
target_link_libraries(cpp PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(cpp PRIVATE superlu_dist)
message("MPI:" ${MPI_C_INCLUDE_DIRS})

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION mpi_solver)