### high-level interation with GPU from Python
"""
from numbapro import vectorize
from numpy import arange

@vectorize(['float32(float32, float32)'], target='gpu') # default to 'cpu'
def add2(a, b):
    return a + b

X = arange(10, dtype='float32')
Y = X * 2
print add2(X, Y)
print add2.reduce(X)
"""



### basic vectorize
"""
import math
from numbapro import vectorize, cuda
import numpy as np

@vectorize(['float32(float32, float32, float32)',
            'float64(float64, float64, float64)'],
           target='gpu')
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)

N = 1e+4
dtype = np.float32

# prepare the input
A = np.array(np.random.sample(N), dtype=dtype)
B = np.array(np.random.sample(N) + 10, dtype=dtype)
C = np.array(np.random.sample(N), dtype=dtype)

D = cu_discriminant(A, B, C)

print(D)  # print result
"""



### NOTE: error code below (about resource allocation):
"""
from numbapro import *
import numpy as np
import math
from timeit import default_timer as time

bpg = 50
tpb = 32
n = bpg * tpb

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:]])
def cu_square_matrix_mul(A, B, C):
    sA = cuda.shared.array(shape=(tpb, tpb), dtype=f4)
    sB = cuda.shared.array(shape=(tpb, tpb), dtype=f4)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    acc = 0.
    for i in range(bpg):
        if x < n and y < n:
            sA[ty, tx] = A[y, tx + i * tpb]
            sB[ty, tx] = B[ty + i * tpb, x]

        cuda.syncthreads()

        if x < n and y < n:
            for j in range(tpb):
                acc += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()

    if x < n and y < n:
        C[y, x] = acc

A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.empty_like(A)

print("N = %d x %d" % (n, n))

s = time()
stream = cuda.stream()
with stream.auto_synchronize():
    dA = cuda.to_device(A, stream)
    dB = cuda.to_device(B, stream)
    dC = cuda.to_device(C, stream)
    cu_square_matrix_mul[(bpg, bpg), (tpb, tpb), stream](dA, dB, dC)
    dC.to_host(stream)

e = time()
tcuda = e - s

# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)

s = time()
Cans = Amat * Bmat
e = time()
tcpu = e - s

print('cpu:  %f' % tcpu)
print('cuda: %f' % tcuda)
print('cuda speedup: %.2fx' % (tcpu / tcuda))

# Check result
assert np.allclose(C, Cans)
"""




'''
This example uses cuBLAS gemm routine to perform matrix-matrix multiplication.
Please refer to the documentation for details of how to use the gemm routine
  http://docs.continuum.io/accelerate/cublas#blas-level-2
Note: cuBLAS uses Fortran layout
'''

from accelerate.cuda.blas import Blas
import numpy as np
from timeit import default_timer as timer

N = 128     # no. of rows/cols

def gemm_v1():
    '''
    Note that all arrays are in Fortran order.
    '''
    print("Version 1".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N), order='F')
    B = np.array(np.arange(N) + 10, dtype=A.dtype, order='F')
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = Blas()

    start = timer()
    blas.gemm('N', 'N', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def gemm_v2():
    """
    Let GEMM transpose the input matrices so that they can be in C order,
    originally.  Note that the output matrix is still in Fortran array.
    The string arguments in gemm tells it to apply transformation on the input
    matrices.
    See argument description in:
        http://docs.continuum.io/accelerate/cublas#blas-level-2
    """
    print("Version 2".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N))
    B = np.array(np.arange(N) + 10, dtype=A.dtype)
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = Blas()

    start = timer()
    blas.gemm('T', 'T', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def main():
    gemm_v1()
    gemm_v2()

if __name__ == '__main__':
   main()


