import numpy as np
import cvxpy as cp
import itertools

datashape = (2,4)
slcs = [(slice(1,2), slice(2,3)), (slice(3,4), slice(1,3))]

def unroll_slice(slices):
    it = itertools.product(*[range(sl.start, sl.stop) for sl in slices])
    return it

print(slcs[0])
amazing = unroll_slice(slcs[0])
for i in amazing:
    print i