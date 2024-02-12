#include "la.h"
#include "superlu.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(cpp, m)
{
  m.doc() = "Python interface";

  m.def("create_mat", []() {});

  nb::class_<MatrixCSR<double>>(m, "MatrixCSR")
      .def(
          "__init__",
          [](MatrixCSR<double>* self,
             nb::ndarray<const double, nb::ndim<1>, nb::c_contig> values,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indptr,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> indices)
          {
            std::vector<std::int64_t> indices_vec(
                indices.data(), indices.data() + indices.size());
            std::vector<std::int32_t> indptr_vec(indptr.data(),
                                                 indptr.data() + indptr.size());
            std::vector<double> values_vec(values.data(),
                                           values.data() + values.size());

            new (self) MatrixCSR<double>(MPI_COMM_WORLD, values_vec, indptr_vec,
                                         indices_vec, indptr_vec.size() - 1);
          });

  m.def("solve",
        [](MatrixCSR<double>& A,
           nb::ndarray<const double, nb::ndim<1>, nb::c_contig> b, bool verbose)
        {
          std::vector<double> u(b.size());
          superlu_solver<double>(A, b.data(), u.data(), verbose);
          return u;
        });
}