import numba
from numba import jit
import numpy as np
import time

from Math import Functions

a = np.random.randn(1000, 1000)
aa = np.random.randn(1000, 1000)

def normal(x):
    for i in range(100):
        result = np.multiply(a, aa)
        result = np.sum(result)

#@jit(nopython=True)
def go_fast(x):  # Function is compiled and runs in machine code

    for i in range(100):
        result = a.ravel().dot(aa.ravel())


# Testing normal execution, without Numba
start = time.time()
normal(aa)
end = time.time()
print("Elapsed normal = %s" % (end - start))

# First iteration of Numba. Numba will have to compile the code during this iteration.
start = time.time()
go_fast(a)
end = time.time()
print("Elapsed gofast = %s" % (end - start))

# 2nd iteration with Numba. Compilation should be complete, this should be a lot faster.
start = time.time()
go_fast(aa)
end = time.time()
print("Elapsed gofast (after Numba compilation) = %s" % (end - start))

# # 3rd iteration with Numba. Compilation should be complete, this should be a lot faster.
# start = time.time()
# go_fast(aa)
# end = time.time()
# print("Elapsed (after Numba compilation) = %s" % (end - start))