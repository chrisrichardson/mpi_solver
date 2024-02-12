
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import functionspace, form
from dolfinx.fem.assemble import assemble_matrix
from ufl import dx, inner, grad, TestFunction, TrialFunction
import numpy as np

mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
Q = functionspace(mesh, ("Lagrange", 1))
u, v = TestFunction(Q), TrialFunction(Q)
a = form(inner(grad(u), grad(v))*dx)

A = assemble_matrix(a)
A.scatter_reverse()

print(A.indptr, A.indices)
im0 = A.index_map(0)
im1 = A.index_map(1)
nr = im0.size_local
nnz = A.indptr[nr]
l2g = im1.local_to_global(np.arange(im1.size_local + im1.num_ghosts))
global_indices = np.array([l2g[i] for i in A.indices[:nnz]])

from cpp import MatrixCSR, solve

A_superlu = MatrixCSR(A.data[:nnz], A.indptr[:nr+1], global_indices)
b = np.ones(len(A.indptr))
u = solve(A_superlu, b, True)
print(u)