
import numpy as np
import cvxpy as cp
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
    for sl_tuple in slices:
        #print("sl tuple is", sl_tuple)
        slice_value = values.pop(0)
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
    for exact_sl_tuple in exact_slices:
        exact_sl_value = exact_values.pop(0)
        unrolled = unroll_slice(exact_sl_tuple)
        running_slice_vars = []
        for index in unrolled:
            lin_index = np.ravel_multi_index(index, datashape)
            running_slice_vars.append(x_arr[lin_index])
        slice_vars_sum = sum(running_slice_vars)
        constraints.append(slice_vars_sum == exact_sl_value)

    # zeros
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

    """
    Given a tuple of slices, returns an iterator that goes through all indices covered by slice
    """
def unroll_slice(slices):

    it = itertools.product(*[range(sl.start, sl.stop) for sl in slices])
    return it

"""
convert to 1-d array (linearize multidimensional index)
np.ravel_multi_index(index, datashape)
to convert back to correct shape use:
X = arr.reshape(datashape)
"""

if __name__ == '__main__':
    shape = (2, 4) #dimensions of numpy array X (output). currently a 2-d with 2 rows and 4 cols
    slcs = [(slice(1,2), slice(2,3)), (slice(0,2), slice(1,4))] #2 dimensions, so each tuple has 2 slices
    vals = [1.0, 2.0] #list of floats, same length as slices list. for each sl in slices, (X[sl].sum() - v)^2 added to obj
                # fxn must introduce a new variable y for each slice so that we add (y-v)^2 to obj fxn and have constraint
                #y == X[sl].sum()
    ex_slcs = [(slice(0,1), slice(3,4))] #list of tuples, same form as slices
    ex_vals = [2.0] # with exact_slices, specifies additional equality constraints. For each sl in exact_values and the
                    # corresponding v from exact_values, we need X[sl].sum() == v
    struct_zeros = [(slice(0,1), slice(1,2))]#list of slices. For each index covered by a slice tuple,
                                           # we need X[index] == 0.

    no_negs = False #if true, all elements in X must be nonnegative

    solve(shape, slcs, vals, ex_slcs, ex_vals, struct_zeros, no_negs)

