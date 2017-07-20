# x is a given numpy array. find y such that:
# minimize: ||x-y||^2
# (1) y is increasing throughout (pairwise)
# (2) last element of y = last element of x
# (3) y[0] >= 0
# x is input_array, y is generate_array

import cvxpy as cp
import numpy as np

#size 3 input for now (input_array is 1-d numpy array)
def min_least_squares(input_array):

    generate_array = cp.Variable(input_array.size)
    increasing = cp.Bool()

    objective = cp.Minimize((cp.sum_entries(cp.square(input_array - generate_array))))



    # haven't figured out how to not hardcode in the 'y is increasing' condition, so we have to put in each
    # <= pair manually for now.
    constraints = [generate_array[-1] == input_array[-1],
                   generate_array[0] <= generate_array[1], generate_array[1] <= generate_array[2],
                   generate_array[0] >= 0]

    print("constraints[0] type is ", type(constraints[0]))
    #constraints = [generate_array[-1] == input_array[-1], generate_array[0] >= 0, increasing == True]

    prob = cp.Problem(objective, constraints)
    prob.solve()  # Returns the optimal value.
    print "status:", prob.status
    print "optimal value", prob.value
    print "optimal var", generate_array.value

if __name__ == '__main__':
    input_array = np.array([5.0, 2.0, 10.0])
    min_least_squares(input_array)


