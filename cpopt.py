
import numpy as np
import cvxpy as cp
import testcvxopt as test
import itertools

#sum of each obj function from slice/value pair expressions


# find a numpy array X having dims "datashape" that minimizes constrained least squares prob specified by other params
# slices and values must have the same length;. exact_slices and exact_values must have the same length
def solve(datashape, slices, values, exact_slices, exact_values, structural_zeros, nnls):
    objective = cp.Minimize(0) #might have to initialize with the first objective expr
    constraints = []
    total_arr_size = 1
    for dimension in datashape:
        total_arr_size *= dimension
    x_arr = [cp.Variable() for i in range(total_arr_size)] #1-d array for working w/solver

    # Loop over slice/val pairs and add a fxn to objective_expressions for each (w/overloaded '+' operator)
    if slices is not None:
        for sl_tuple in slices:
            #print("sl tuple is", sl_tuple)
            slice_value = values[0]
            values = np.delete(values, 0)
            unrolled = unroll_slice(sl_tuple)
            running_slice_vars = []
            for index in unrolled:
                #print("index is", index)
                lin_index = np.ravel_multi_index(index, datashape)
                #print("lin_index is", lin_index)
                running_slice_vars.append(x_arr[lin_index])
                #print("running_slice_vars is", running_slice_vars)

            slice_vars_sum = sum(running_slice_vars)
            objective += cp.Minimize((slice_vars_sum - slice_value)**2)

    # equality constraints
    if exact_slices is not None:
        for exact_sl_tuple in exact_slices:
            exact_sl_value = exact_values[0]
            exact_values = np.delete(exact_values, 0)
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
                print("assigning", lin_index, "as zero")
                constraints.append(x_arr[lin_index] == 0)

    # no negatives
    if(nnls):
        for var in x_arr:
            constraints.append(var >= 0)

    to_solve = cp.Problem(objective, constraints)
    to_solve.solve()  # Returns the optimal value.
    print "status:", to_solve.status
    print "optimal value", to_solve.value
    print "optimal return array values:"
    for var in x_arr:
        print var.value
    #print "optimal var", x_arr.value
    np.reshape(x_arr, datashape)

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

    first = test.Hier.solve(data.shape, slices, b)

    print("first is:", first)

