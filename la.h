#include <array>
#include <iostream>
#include <mpi.h>
#include <vector>

#pragma once

template <typename T>
class MatrixCSR
{
public:
  MatrixCSR(MPI_Comm mat_comm, const std::vector<T>& mat_data,
            const std::vector<std::int32_t>& mat_indptr,
            const std::vector<std::int64_t>& mat_indices,
            std::int64_t num_local_rows)
      : comm(mat_comm), data(mat_data), indptr(mat_indptr), indices(mat_indices)
  {
    MPI_Allreduce(&num_local_rows, &global_num_rows, 1, MPI_INT64_T, MPI_SUM,
                  comm);
    std::cout << "local num rows = " << num_local_rows << "\n";

    local_row_range[0] = 0;
    MPI_Exscan(&num_local_rows, &local_row_range[0], 1, MPI_INT64_T, MPI_SUM,
               comm);
    std::cout << "range = " << local_row_range[0] << "-" << local_row_range[1]
              << "\n";
    local_row_range[1] = local_row_range[0] + num_local_rows;
  }

  MPI_Comm comm;
  // Data values at non-zeros
  std::vector<T> data;
  // Offsets for each local row
  std::vector<std::int32_t> indptr;
  // Column indices on each local row (global index)
  std::vector<std::int64_t> indices;

  std::array<std::int64_t, 2> local_row_range;
  std::int64_t global_num_rows;
};
