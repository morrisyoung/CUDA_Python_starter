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


**Second thought**: Matrix multiplication is mostly for gradient descent style solvers. For sampling based ones like Gibbs sampling (though also involving some MM), I'm not sure whether GPU could help, since we might be able to sample independent parameters simultaneously. So, to what extend can we identify these independent parameters?


**Third thought**: Python CUDA might help for Variational Inference algorithms, since the most usual solver is coordinate ascent style, in which we iteratively take numerical calculation (not sampling, though similar to Gibbs sampling) for each of the variational parameters. However, since VI 1. is already an approximation to the true posterior inference (without theoretical guarantee), and 2. is already very fast (each faster for stochastic variational inference), I'm not sure it's worth our efforts to transform the code into GPU code in order to do VI faster. Also, according to what we saw above, it's already very fast if the code is optimized properly with numpy.


