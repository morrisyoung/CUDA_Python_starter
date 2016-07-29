Run samples from [NumbaPro](https://docs.continuum.io/numbapro/index-archived) and [reference examples](https://github.com/ContinuumIO/numbapro-examples). cuBLAS is said to be fast for matrix multiplication (see [here](https://github.com/ContinuumIO/numbapro-examples/tree/master/cublas)), while still not that much, and sometimes even slower than optimized Numpy code. So I won't use NumbaPro (Accelerate) to optimize Numpy code.

Result from [test_mm.py](https://github.com/morrisyoung/CUDA_Python_starter/blob/master/test_mm.py) for different methods of Matrix Multiplication and their speed:

case#1:
```
n x n = 1024 x 1024
Using numpy.dot: 0.07 s
Without shared memory: 0.26 s
With shared memory: 0.13 s
Using CuBLAS: 0.22 s

```

case#2:
```
n x n = 2048 x 2048
Using numpy.dot: 0.16 s
Without shared memory: 1.82 s
With shared memory: 1.06 s
Using CuBLAS: 0.27 s
```
