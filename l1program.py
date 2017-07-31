
import numpy as np
import cvxpy as cp
import itertools

#lists must be of equal length
def calc_pairwise_dist(list1, list2):
    diffs = [list[i] - list2[i] for i in range(len(list1))]
    return sum(diffs)

#Minimize obj_vector's L1 distance from input_vector subject to:
#(1) obj_vector[0] = 0
#(2) obj_vector is increasing throughout
#(3) obj_vector is nonnegative throughout
#(4) sum of values in obj_vector = an input, total_unit_count
def solve(input_vector, total_unit_count):
    obj_var = cp.Variable()
    obj_vector = []
    constraints = []
    for i in range(input_vector.size):
        var = cp.Int()
        obj_vector.append(var)
        constraints.append(var >= 0)
        if i != 0:
            constraints.append(var >= obj_vector[i-1])

    constraints.append(obj_vector[0] == 0)
    constraints.append(obj_var >= calc_pairwise_dist(input_vector, obj_vector))
    constraints.append(obj_var >= calc_pairwise_dist(obj_vector, input_vector))

    objective = cp.Minimize(obj_var)

    to_solve = cp.Problem(objective, constraints)
    to_solve.solve(solver=cp.CVXOPT)
    print "status:", to_solve.status
    print "optimal value", to_solve.value
    print "optimal return array values:"
    #for var in x_arr:
       # print var.value
    #print "optimal var", x_arr.value



if __name__ == '__main__':
    prefix_sums = []
    unitcount = 0
    solve(prefix_sums, unitcount)

