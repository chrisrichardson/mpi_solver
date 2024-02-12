
#pragma once

#include "la.h"
#include <mpi.h>
#include <vector>

/// Solve A.u = b with SuperLU_dist
/// @param comm MPI_Comm
/// @param Amat CSR Matrix, distributed by row and finalized
/// @param bvec RHS vector
/// @param uvec Solution vector
/// @param verbose Output diagnostic information to stdout
template <typename T>
int superlu_solver(const MatrixCSR<T>& Amat, const T* bvec, T* uvec,
                   bool verbose = true);
