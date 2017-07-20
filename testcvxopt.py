from __future__ import division,print_function
########
import sys
import os
#sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
########
import cpopt as cp
import numpy
import unittest
import itertools
import logging
from scipy.sparse.linalg import LinearOperator, lsqr
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import lsq_linear

class TestCpopt(unittest.TestCase):
    def test_solve(self):
        data = numpy.random.random_integers(0,100, (2,3))
        point_slices = [(slice(i,i+1), slice(j, j+1)) for i in range(2) for j in range(3)]
        slices = [(slice(0,1), slice(0,2)),
                  (slice(0,1), slice(1,3)),
                  (slice(1,2), slice(0,2)),
                  (slice(1,2), slice(1,3)),
                  (slice(0,2), slice(0,1)),
                  (slice(0,2), slice(1,2)),
                  (slice(0,2), slice(2,3))] + point_slices
        b = numpy.random.random_integers(-2, 2, size=len(slices)) + numpy.array([data[x].sum() for x in slices])
        logging.debug(data)
        logging.debug(b)
        first = Hier.solve(data.shape, slices, b)
        logging.debug(first)
        second = cp.solve(data.shape, slices, b, [], [], structural_zeros=None, nnls=False)
        firsterrors = sum([(numpy.sum(first[sl])-y)**2 for (sl, y) in zip(slices,b)])
        seconderrors = sum([(numpy.sum(second[sl])-y)**2 for (sl, y) in zip(slices,b)])
        print("Expected result:", first,"result From cvx:", second)
        print("Expected error:", firsterrors, "cvx error:", seconderrors)
        self.assertTrue(abs(firsterrors - seconderrors) < 0.000001)        
        #print(first, second)
        #print(firsterrors, seconderrors)
        #self.assertTrue(abs(firsterrors - seconderrors) < 0.000001)

    def test_nnls(self):
        #numpy.random.seed(43)
        data = numpy.random.random_integers(-100,100, (2,3)) + 0.0
        point_slices = [(slice(i,i+1), slice(j, j+1)) for i in range(2) for j in range(3)]
        slices = [(slice(0,1), slice(0,2)),
                  (slice(0,1), slice(1,3)),
                  (slice(1,2), slice(0,2)),
                  (slice(1,2), slice(1,3)),
                  (slice(0,2), slice(0,1)),
                  (slice(0,2), slice(1,2)),
                  (slice(0,2), slice(2,3))] + point_slices
        b = numpy.random.random_integers(-2, 2, size=len(slices)) + numpy.array([data[x].sum() for x in slices]) + 0.0
        logging.debug(data)
        logging.debug(b)
        first = Hier.nnls_lbfgs_b(data.shape, slices, b)
        logging.debug(first)
        second = cp.solve(data.shape, slices, b, [], [], structural_zeros=None,  nnls=True)
        firsterrors = sum([(numpy.sum(first[sl])-y)**2 for (sl, y) in zip(slices,b)])
        seconderrors = sum([(numpy.sum(second[sl])-y)**2 for (sl, y) in zip(slices,b)])
        print("expected result:" + str(first))
        print("result from cvx:" + str(second))
        print("expectederrors:", firsterrors, "cvxerrors:", seconderrors)
        self.assertTrue(abs(firsterrors - seconderrors)/(1.0+firsterrors) < 0.000001)
        
    def atest_nnls_weighted(self):
        data = numpy.random.random_integers(-100,100, (2,3))
        point_slices = [(slice(i,i+1), slice(j, j+1)) for i in range(2) for j in range(3)]
        slices = [(slice(0,1), slice(0,2)),
                  (slice(0,1), slice(1,3)),
                  (slice(1,2), slice(0,2)),
                  (slice(1,2), slice(1,3)),
                  (slice(0,2), slice(0,1)),
                  (slice(0,2), slice(1,2)),
                  (slice(0,2), slice(2,3))] + point_slices
        b = numpy.random.random_integers(-2, 2, size=len(slices)) + numpy.array([data[x].sum() for x in slices])
        weights = numpy.random.uniform(1,2,b.size)
        logging.debug(data)
        logging.debug(b)
        first = Hier.nnls_lbfgs_b(data.shape, slices, b, weights=weights)
        logging.debug(first)
        second = cp.solve(data.shape, slices, b, [], [], structural_zeros=None, nnls=True, weights=weights)
        firsterrors = sum([(numpy.sum(first[sl])-y)**2 for (sl, y) in zip(slices,b)])
        seconderrors = sum([(numpy.sum(second[sl])-y)**2 for (sl, y) in zip(slices,b)])
        print("Expected result:" + str(first))
        print("Result From cvx:" + str(second))
        print("expectederrors:", firsterrors, "cvxerrors:", seconderrors)
        self.assertTrue(abs(firsterrors - seconderrors)/(1.0+firsterrors) < 0.00001)
        
    def test_solve_with_exact(self):
        data_shape = (1,4)
        slices = [(slice(0,1), slice(0,1)), 
                  (slice(0,1), slice(1,2)),
                  (slice(0,1), slice(2,3)),
                  (slice(0,1), slice(3,4)),
                  (slice(0,1), slice(2,4))]
        values = [1, 2, 3, 4, 7]
        exact_slices = [(slice(0,1), slice(0,2))]
        exact_values = [2]
        first = numpy.array([[0.5, 1.5, 3.0, 4.0]])
        logging.debug(first)
        ##problem 103
        second = cp.solve(data_shape, slices, values, exact_slices, exact_values, structural_zeros=None, nnls=False)
        ##problem 103
        firsterrors = sum([(numpy.sum(first[sl])-y)**2 for (sl, y) in zip(slices, values)])
        seconderrors = sum([(numpy.sum(second[sl])-y)**2 for (sl, y) in zip(slices,values)])
        print("Expected result:", first,"result From cvx:", second)
        print("Expected error:", firsterrors, "cvx error:", seconderrors)
        self.assertTrue(abs(firsterrors - seconderrors) < 0.000001)        


class Hier(LinearOperator):
    ###def __init__(self, data_shape, slices, rev_slice, weights=None):
    def __init__(self, data_shape, slices, weights=None):
        self.data_shape = data_shape
        self.slices = slices
        ###self.rev_slice = rev_slice
        self.dtype = numpy.float64
        self.sqweights = numpy.sqrt(weights) if weights is not None else numpy.ones(len(self.slices))

    @property
    def shape(self):
        return (len(self.slices), numpy.prod(self.data_shape))

    def _matvec(self, v):
        synth = v.reshape(self.data_shape)
        ans = numpy.array([float(synth[sl].sum()) for sl in self.slices])
        if self.sqweights is not None:
            ans = ans * self.sqweights
        return ans


    def _rmatvec(self, v):
        #ans = None
        #if self.sqweights is None:
        #    ans = numpy.array([v[sl].sum() for sl in self.rev_slice])
        #else:
        #    ans = numpy.array([(v[sl] * self.sqweights[sl]).sum() for sl in self.rev_slice])
        #return ans
        ans = numpy.zeros(self.data_shape)
        for (element, sl, w) in itertools.izip(v, self.slices, self.sqweights):
            ans[sl] += element * w
        return ans

    @staticmethod
    ###def solve(data_shape, slices, rev_slice, b, weights=None):
    ###    linop = Hier(data_shape, slices, rev_slice, weights)
    def solve(data_shape, slices, b, weights=None, method=None, exact_slices=None, exact_b=None):
        answer = None
        linop = Hier(data_shape, slices, weights)
        y = b if weights is None else b * numpy.sqrt(weights)
        result = lsqr(linop, y)
        answer = result[0].reshape(data_shape)
        #TODO add exact slices
        return answer

    @staticmethod
    def nn_ipa(synth, slices=None, values=None, iters=30):
        """ set negative counts to 0 and then
        use iterative proportional fitting to get synth to satisfy the linear
        constraints synth[x].sum() = v for (x,v) in zip(slices, values) """
        if slices is None or len(slices) == 0:
            slices = [tuple([slice(0,x) for x in synth.shape])]
            values = [synth.sum()]
        mydata = numpy.where(synth > 0.0, synth, 0.0)
        for _ in range(iters):
            for sl, v in zip(slices, values):
                ans = mydata[sl].sum()
                mydata[sl] *= v/float(ans)
        return mydata

    @staticmethod
    def l2_nonneg_tot(data, tot = None):
        """ find nonnegative histogram closest to data in L2 sense while also having same sum """
        if tot is None:
            tot = data.sum()
        arr = data.astype(numpy.float64).reshape(data.size)
        mydata = data.astype(numpy.float64)
        arr.sort()
        the_min = arr.min()
        arr = arr - the_min
        mydata = mydata - the_min
        curr_tot = arr.sum()
        if(curr_tot <= tot):
            mydata += (tot-curr_tot)/mydata.size
        else:
            cutoff = 0
            remainder = 0.0
            cumul = arr.copy()
            cumul[1:] = numpy.diff(arr)
            cumul = curr_tot - tot - (cumul * numpy.arange(mydata.size,0,step=-1)).cumsum()
            i = 0
            while cumul[i] >= 0:
                i = i + 1
            cutoff = i-1
            remainder = cumul[cutoff]
            nonzeros = numpy.where(mydata <= arr[cutoff], 0, 1).sum()
            mydata = numpy.where(mydata <= arr[cutoff], 0, mydata - arr[cutoff]- remainder/float(nonzeros))
        return mydata


    @staticmethod
    ###def nnls_lbfgs_b(data_shape, slices, rev_slice, b, weights=None, guess=None):
    ###    linop = Hier(data_shape, slices, rev_slice, weights)
    def nnls_lbfgs_b(data_shape, slices,  b, weights=None, guess=None):
        linop = Hier(data_shape, slices,  weights)
        y = b if weights is None else b * numpy.sqrt(weights)
        def loss_and_grad(x):
            residual = linop.matvec(x) - y
            grad = linop.rmatvec(residual)
            loss = 0.5 * numpy.sum(residual ** 2)
            return loss, grad
        x_dim = linop.shape[1]
        if guess is None:
            guess = numpy.zeros(x_dim)
        bounds = [(0, None)] * x_dim
        result = fmin_l_bfgs_b(loss_and_grad, x0=guess, pgtol=0.00001, \
                               bounds=bounds, maxiter=1000, m=1)
        logging.debug(result[2])
        return result[0].reshape(data_shape)
        #TODO: add exact constraints?


    @staticmethod
    def trf_nnls(data_shape, slices, b, weights = None):
        linop = Hier(data_shape, slices,  weights)
        y = b if weights is None else b * numpy.sqrt(weights)
        x_dim = linop.shape[1]
        bounds = (0, numpy.inf)
        result = lsq_linear(linop,y,bounds=bounds,method='trf',max_iter=1000, tol=0.0001)
        logging.debug(str(result))
        return result[0].reshape(data_shape)

    @staticmethod
    def ls_and_nnls(data_shape, slices, b, weights=None, method='l_bfgs_b', exact_slices=None, exact_values=None, ls_method=None):
        pid = str(os.getpid())
        logging.debug("%s starting ls ..." % (pid,))
        ###ls = Hier.solve(data_shape, slices, rev_slice, b, weights)
        ls = Hier.solve(data_shape, slices, b, weights, method=ls_method)
        logging.debug("%s ending ls" % (pid,))
        nnls = None
        if method is not None:
            logging.debug("%s starting nnls ..." % (pid,))
            if method == 'l_bfgs_b':
                uniform = numpy.prod(ls.shape) / ls.sum()
                nnls = Hier.nnls_lbfgs_b(data_shape, slices, b, weights, numpy.zeros(ls.shape)+uniform)
            elif method == 'trf':
                nnls = Hier.trf_nnls(data_shape, slices=slices, b=b, weights = weights)
            elif method == 'simple':
                nnls = Hier.l2_nonneg_tot(ls)
            elif method == 'ipa':
                nnls = Hier.nn_ipa(ls)
            elif method == 'cut':
                total = ls.sum()
                nnls = ls.copy()
                nnls[nnls< 0]=0
                indices = numpy.where(nnls>0)
                tmp_nnls = Hier.l2_nonneg_tot(nnls[indices], tot=total)
                nnls[indices]=tmp_nnls
            elif method == 'smartcut':
                total = ls.sum()
                nnls = ls.copy()
                lowest = min(0, nnls.min())
                thresh = numpy.abs(lowest) + 0.001
                nnls[nnls < thresh]=0
                indices = numpy.where(nnls>0)
                tmp_nnls = Hier.l2_nonneg_tot(nnls[indices], tot=total)
                nnls[indices]=tmp_nnls
            else:
                message = "unknown nnls method" + method
                logging.debug(message)
                raise Exception(message)
            nnls[nnls<0]=0
            logging.debug("%s ending nnls" % (pid,))
        return ls, nnls
        #TODO add exact contraints



if __name__ == "__main__":
    logging.basicConfig(filename= "test_cvxopt.log",format='%(levelname)s:%(asctime)s [%(process)d] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCpopt)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()
