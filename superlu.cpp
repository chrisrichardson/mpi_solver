
#include "superlu.h"
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_zdefs.h"
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
int superlu_solver(const MatrixCSR<T>& Amat, const T* bvec, T* uvec,
                   bool verbose)
{
  MPI_Comm comm = Amat.comm;
  int size;
  MPI_Comm_size(comm, &size);

  int nprow = size;
  int npcol = 1;

  gridinfo_t grid;
  superlu_gridinit(comm, nprow, npcol, &grid);

  // Global size
  int m = Amat.global_num_rows;
  int n = Amat.global_num_rows;

  // Number of local rows
  int m_loc = Amat.local_row_range[1] - Amat.local_row_range[0];

  // First row
  int first_row = Amat.local_row_range[0];

  std::cout << "first row = " << first_row << "\n";
  std::cout << "m_loc = " << m_loc << "\n";

  // Local number of non-zeros
  int nnz_loc = Amat.indptr[m_loc];
  std::vector<int_t> cols(nnz_loc);
  std::vector<int_t> rowptr(m_loc + 1);

  // Copy row_ptr to int_t
  std::copy(Amat.indptr.begin(), std::next(Amat.indptr.begin(), m_loc + 1),
            rowptr.begin());

  // Copy column indices to int_t
  std::copy(Amat.indices.begin(), std::next(Amat.indices.begin(), nnz_loc),
            cols.begin());

  SuperMatrix A;
  auto Amatdata = const_cast<T*>(Amat.data.data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   reinterpret_cast<doublecomplex*>(Amatdata),
                                   cols.data(), rowptr.data(), SLU_NR_loc,
                                   SLU_Z, SLU_GE);
  }

  // RHS
  int ldb = m_loc;
  int nrhs = 1;

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;
  if (!verbose)
    options.PrintStat = NO;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU replaces RHS with solution)
  std::copy(bvec, bvec + m_loc, uvec);

  if constexpr (std::is_same_v<T, double>)
  {
    std::vector<T> berr(nrhs);
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    pdgssvx(&options, &A, &ScalePermstruct, uvec, ldb, nrhs, &grid, &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
    dSolveFinalize(&options, &SOLVEstruct);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    std::vector<T> berr(nrhs);
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sScalePermstructInit(m, n, &ScalePermstruct);
    sLUstructInit(n, &LUstruct);
    sSOLVEstruct_t SOLVEstruct;

    psgssvx(&options, &A, &ScalePermstruct, uvec, ldb, nrhs, &grid, &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    sSolveFinalize(&options, &SOLVEstruct);
    sLUstructFree(&LUstruct);
    sScalePermstructFree(&ScalePermstruct);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    std::vector<double> berr(nrhs);
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);
    zSOLVEstruct_t SOLVEstruct;

    pzgssvx(&options, &A, &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(uvec), ldb, nrhs, &grid, &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
    zSolveFinalize(&options, &SOLVEstruct);
  }
  Destroy_SuperMatrix_Store_dist(&A);

  if (info)
  {
    std::cout << "ERROR: INFO = " << info << " returned from p*gssvx()\n"
              << std::flush;
  }

  if (verbose)
    PStatPrint(&options, &stat, &grid);
  PStatFree(&stat);

  superlu_gridexit(&grid);

  return info;
}

// Explicit instantiation
template int superlu_solver(const MatrixCSR<double>&, const double*, double*,
                            bool);

template int superlu_solver(const MatrixCSR<float>&, const float*, float*,
                            bool);

template int superlu_solver(const MatrixCSR<std::complex<double>>&,
                            const std::complex<double>*, std::complex<double>*,
                            bool);
