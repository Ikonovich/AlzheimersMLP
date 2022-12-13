import numba
from numba import jit
import numpy as np
import time

from Math import Functions

a = np.random.randn(1000, 1000).flatten()
aa = np.random.randn(1000, 1000).flatten()

def normal(x):
    return np.maximum(x, 0)


@jit(nopython=True)
def go_fast(x):  # Function is compiled and runs in machine code
    return np.maximum(x, 0)

print(str(numba.typeof(Functions.sigmoid_two)))

# # Testing normal execution, without Numba
# start = time.time()
# normal(aa)
# end = time.time()
# print("Elapsed (without Numba) = %s" % (end - start))
#
# # First iteration of Numba. Numba will have to compile the code during this iteration.
# start = time.time()
# go_fast(a)
# end = time.time()
# print("Elapsed (with Numba compilation) = %s" % (end - start))
#
# # 2nd iteration with Numba. Compilation should be complete, this should be a lot faster.
# start = time.time()
# go_fast(aa)
# end = time.time()
# print("Elapsed (after Numba compilation) = %s" % (end - start))
#
# # 3rd iteration with Numba. Compilation should be complete, this should be a lot faster.
# start = time.time()
# go_fast(aa)
# end = time.time()
# print("Elapsed (after Numba compilation) = %s" % (end - start))