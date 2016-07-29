##
## from: https://people.duke.edu/~ccc14/sta-663/CUDAPython.html#more-examples
##

from numbapro import cuda, vectorize, guvectorize, check_cuda
from numbapro import void, uint8 , uint32, uint64, int32, int64, float32, float64, f8
import numpy as np
from timeit import default_timer as timer
import numbapro.cudalib.cublas as cublas



check_cuda()
device = cuda.get_current_device()




### naive matrix multiplication
"""
x1 = np.random.random((4,4))
x2 = np.random.random((4,4))
np.dot(x1, x2).shape
"""



### Kernel function (no shared memory)
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], int32)')
def cu_matmul(a, b, c, n):
    x, y = cuda.grid(2)

    if (x >= n) or (y >= n):
        return

    c[x, y] = 0
    for i in range(n):
        c[x, y] +=  a[x, i] * b[i, y]

"""
tpb = device.WARP_SIZE
n = 400
bpg = (n+tpb-1)//tpb
grid_dim = (bpg, bpg)
block_dim = (tpb, tpb)

A = np.random.random((n, n)).astype(np.float32)
B = np.random.random((n, n)).astype(np.float32)
C = np.empty((n, n), dtype=np.float32)
cu_matmul[grid_dim, block_dim](A, B, C, n)
assert(np.allclose(np.dot(A, B), C))
"""





### Kernel function (with shared memory)
tpb = device.WARP_SIZE
block_dim = (tpb, tpb)

@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], int32, int32, int32)')
def cu_matmul_sm(A, B, C, n, tpb, bpg):
    # decalre shared memory
    sA = cuda.shared.array(shape=block_dim, dtype=float32)
    sB = cuda.shared.array(shape=block_dim, dtype=float32)

    # we now need the thread ID within a block as well as the global thread ID
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x, y = cuda.grid(2)

    # pefort partial operations in block-szied tiles
    # saving intermediate values in an accumulator variable
    acc = 0.0
    for i in range(bpg):
        # Stage 1: Prefil shared memory with current block from matrix A and matrix B
        sA[tx, ty] = A[x, ty + i * tpb]
        sB[tx, ty] = B[tx + i * tpb, y]

        # Block calculations till shared mmeory is filled
        cuda.syncthreads()

        # Stage 2: Compute partial dot product and add to accumulator
        if x < n and y < n:
            for j in range(tpb):
                acc += sA[tx, j] * sB[j, ty]

        # Blcok until all threads have completed calcuaiton before next loop iteration
        cuda.syncthreads()

    # Put accumulated dot product into output matrix
    if x < n and y < n:
        C[x, y] = acc

"""
k = 32
n = tpb * k # n must be multiple of tpb because shared memory is not initialized to zero
bpg = n//tpb
grid_dim = (bpg, bpg)

A = np.random.random((n, n)).astype(np.float32)
B = np.random.random((n, n)).astype(np.float32)
C = np.empty((n, n), dtype=np.float32)
cu_matmul_sm[grid_dim, block_dim](A, B, C, n, tpb, bpg)
assert(np.allclose(np.dot(A, B), C))
"""





### Benchmark
k = 64		# NOTE: tunable
n = tpb * k
bpg = n//tpb
grid_dim = (bpg, bpg)

# Prepare data on the CPU
A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.zeros_like(A)

print "n x n = %d x %d" % (n, n)

# Prepare data on the GPU
dA = cuda.to_device(A)
dB = cuda.to_device(B)
dC = cuda.to_device(C) # device_array_like(A)

# Time numpy version
s = timer()
np_ans = np.dot(A, B)
e = timer()
t = e - s

# Time the unoptimized version
s = timer()
cu_matmul[grid_dim, block_dim](dA, dB, dC, n)
cuda.synchronize()
e = timer()
unopt_ans = dC.copy_to_host()
tcuda_unopt = e - s

# Time the shared memory version
s = timer()
cu_matmul_sm[grid_dim, block_dim](dA, dB, dC, n, tpb, bpg)
cuda.synchronize()
e = timer()
opt_ans = dC.copy_to_host()
tcuda_opt = e - s

# Time for CuBLAS version
s = timer()
blas = cublas.Blas()
blas.gemm('T', 'T', n, n, n, 1.0, A, B, 1.0, C) # A, B not in fortran order so need for transpose
e = timer()
blas_ans = dC.copy_to_host()
tcuda_blas = e - s

print "Using numpy.dot:", "%.2f" % t, "s"
print "Without shared memory:", "%.2f" % tcuda_unopt, "s"
print "With shared memory:", "%.2f" % tcuda_opt, "s"
print "Using CuBLAS:", "%.2f" % tcuda_blas, "s"



print np.allclose(np_ans, unopt_ans)
print np.allclose(np_ans, opt_ans)
print np.allclose(np_ans, blas_ans)



