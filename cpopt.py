import logging
import numpy as np
import cvxpy as cp
import testcvxopt as test
import itertools


# find a numpy array X having dims "datashape" that minimizes constrained least squares prob specified by other params
# slices and values must have the same length;. exact_slices and exact_values must have the same length
# solver_params is a length 4 list of the following form: [int, double, double, double], where int is maximum iterations
#double 1 is absolute tolerance, double 2 is relative tolerance, double 3 is feasibility tolerance
def solve(datashape, slices, values, exact_slices, exact_values, nnls=False, weights=None, structural_zeros=None, solver_params=(40, 1e-7, 1e-6, 1e-7)):
    objective = cp.Minimize(0)
    constraints = []
    total_arr_size = np.prod(datashape)
    x_arr = [cp.Variable() for i in range(total_arr_size)] #1-d array for working w/solver

    if weights is None:
        weights = np.ones(len(slices))
    if slices is not None:
        for (sl_tuple, slice_value, wgt) in zip(slices, values, weights):
            unrolled = unroll_slice(sl_tuple)
            running_slice_vars = []
            for index in unrolled:
                lin_index = np.ravel_multi_index(index, datashape)
                running_slice_vars.append(x_arr[lin_index])
            slice_vars_sum = sum(running_slice_vars)

            placeholder_var = cp.Variable()
            constraints.append(placeholder_var - slice_vars_sum == 0)
            objective += cp.Minimize(wgt * (placeholder_var - slice_value)**2)

    # equality constraints
    if exact_slices is not None:
        for (exact_sl_tuple, exact_sl_value) in zip(exact_slices, exact_values):
            unrolled = unroll_slice(exact_sl_tuple)
            running_slice_vars = []
            for index in unrolled:
                lin_index = np.ravel_multi_index(index, datashape)
                running_slice_vars.append(x_arr[lin_index])
            slice_vars_sum = sum(running_slice_vars)
            constraints.append(slice_vars_sum == exact_sl_value)

    # zeros
    if structural_zeros is not None:
        for zero_sl_tuple in structural_zeros:
            unrolled = unroll_slice(zero_sl_tuple)
            for index in unrolled:
                lin_index = np.ravel_multi_index(index, datashape)
                #print("assigning", lin_index, "as zero")
                constraints.append(x_arr[lin_index] == 0)

    # no negatives
    if(nnls):
        for var in x_arr:
            constraints.append(var >= 0)

    iters = solver_params[0]
    abs_tolerance = solver_params[1]
    rel_tolerance = solver_params[2]
    feas_tolerance = solver_params[3]

    to_solve = cp.Problem(objective, constraints)
    to_solve.solve(solver=cp.CVXOPT, max_iters=iters, abstol=abs_tolerance, reltol=rel_tolerance, feastol=feas_tolerance)
    logging.debug("status: {}".format(to_solve.status))
    logging.debug("optimal value: {}".format(to_solve.value))
    #print "optimal return array values:"
    #for var in x_arr:
    #    print var.value
    #print "optimal var", x_arr.value
    answer = np.reshape(x_arr, datashape)
    return answer

"""
Given a tuple of slices, returns an iterator that goes through all indices covered by slice
"""
def unroll_slice(slices):

    it = itertools.product(*[range(sl.start, sl.stop) for sl in slices])
    return it


if __name__ == '__main__':
    data = np.random.random_integers(0,100, (2,3))
    point_slices = [(slice(i,i+1), slice(j, j+1)) for i in range(2) for j in range(3)]
    slices = [(slice(0,1), slice(0,2)),
              (slice(0,1), slice(1,3)),
              (slice(1,2), slice(0,2)),
              (slice(1,2), slice(1,3)),
              (slice(0,2), slice(0,1)),
              (slice(0,2), slice(1,2)),
              (slice(0,2), slice(2,3))] + point_slices
    b = np.random.random_integers(-2, 2, size=len(slices)) + np.array([data[x].sum() for x in slices])
    #struct_zeros = [(slice(0,1), slice(1,2))]#list of slices. For each index covered by a slice tuple,
    # we need X[index] == 0.


    #no_negs = False #if true, all elements in X must be nonnegative

    solve(data.shape, slices, b, [], [], structural_zeros = None, nnls= False)


