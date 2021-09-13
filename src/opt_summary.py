from del_summary import *
from numba import njit


class OptSummary(DeletionSummary):
    # Implementation of the delete while summarizing methods described in the paper.


    def __init__(self, values, f, beta, k, method='con'):
        DeletionSummary.__init__(self, values, f, beta, k)
        self.M = np.zeros((len(f), beta, k + 1)) # Matrix that stores the error values. M[q, b, k] = optimal error with b buckets and k deletions when we have searched until the index q
        self.L = np.zeros((len(f), beta, k + 1)) # Matrix used to store the left edge of the rightmost bucket
        self.R = np.zeros((len(f), beta, k + 1)) # Matrix used to store the right edge of the rightmost bucket
        self.lme = np.zeros((len(f), len(f), k + 1)) # Matrix that stores the precomputed lower_max_error values needed for an optimization in the 'arb' method

        self.method = method # used to specify which method to use
        # 'con' for consistent deletions
        # 'arb' for arbitary deletions

        # Flag used to test the jit compile optimization
        self.jit = False

        # if 'arb' is specified precompute the self.lme matrix
        if method == 'arb':
            self.preLowerMaxErr()

    def summarize(self, k):
        r = self.MultiBucketErr(k)
        return r

    def MultiBucketErr(self, k):
        # Find the optimal summary that allows for k deletions.
        # using the delete while summarizing strategy.
        # Implemented according to Figure 5(Arb), Figure 6(Con) in the paper.
        # Depending on self.method applies the method selected.
        # k = int, number of deletions.
        # returns self.M[len(self.f)-1, self.beta-1 , k] that is the minimum error covering
        # the whole dataset with beta buckets and k deletions.

        minl = 0
        minr = 0
        k_opt = 0
        for k_prime in range(k + 1):
            for p in range(1, len(self.f)):
                e_min = float('inf')
                if self.method == 'con':
                    self.M[p, 0, k_prime] , opt_l, opt_r, _= self.conSingleBucketError(0, p, k_prime)
                    self.L[p, 0, k_prime] = opt_l # left edge of rightmost bucket
                    self.R[p, 0, k_prime] = opt_r # right edge of rightmost bucket
                elif self.method == 'arb':
                    self.M[p, 0, k_prime] = self.arbSingleBucketError(0, p, k_prime)
                else:
                    print("Wrong method provided.")
                    return -1

        for b_prime in range(1, self.beta):
            for p in range(1, len(self.f)):
                for k_prime in range(k + 1):
                    if self.method == 'con':
                        self.conUpdateMatrix(p, k_prime, b_prime)
                    elif self.method == 'arb':
                        self.arbUpdateMatrix(p, k_prime, b_prime)
                    else:
                        print("Wrong method provided.")
                        return -1

        return self.M[len(self.f)-1, self.beta-1 , k]


    def arbUpdateMatrix(self, p, k, b):
        # Sub procedure used to update the error matrix M.
        # Implemented according to Figure 5 in the paper.
        # p = int, index of the self.values and self.f arrays (frequency vector)
        # b = int, bucket
        # k = int, number of deletions

        if self.f[p] <= k:
            self.M[p, b, k] = self.M[p - 1, b, k-self.f[p]]
        else:
            self.M[p, b, k] = float(np.inf)
        for q in range(p):
            for k_prime in range(k + 1):
                e = self.M[q, b - 1, k - k_prime]
                e_prime = self.lme[q + 1, p, k_prime]
                self.M[p, b, k] = np.fmin(self.M[p, b, k], e + e_prime)


    @staticmethod
    @njit
    def conUpdateMatrix_(M, p, k, b, f, sqerror, L , R):
        # a static helper function that is called instead of conUpdateMatrix
        # when self.jit = True, to test the speedup
        if f[p] <= k:
            M[p, b, k] = M[p - 1, b, k-f[p]]
            L[p, b, k] = L[p - 1, b, k-f[p]] # left edge of rightmost bucket
            R[p, b, k] = R[p - 1, b, k-f[p]] # right edge of rightmost bucket
        else:
            M[p, b, k] = float(np.inf)

        for q in range(p - 1, -1, -1):
            e = M[q, b - 1, k] + sqerror[q + 1, p]
            if e < M[p, b, k]:
                L[p, b, k] = q + 1 # left edge of rightmost bucket
                R[p, b, k] = p # right edge of rightmost bucket
                M[p, b, k] = e
        return M, L , R

    def conUpdateMatrix(self, p, k, b):
        # Sub procedure used to update the error matrix M.
        # Implemented according to Figure 6 in the paper.
        # p = int, index of the self.values and self.f arrays (frequency vector)
        # b = int, bucket
        # k = int, number of deletions

        if self.jit == True:
            self.M, self.L, self.R= self.conUpdateMatrix_(self.M, p, k, b, self.f, self.sqerror, self.L, self.R)
            return

        if self.f[p] <= k:
            self.M[p, b, k] = self.M[p - 1, b, k-self.f[p]]
            self.L[p, b, k] = self.L[p - 1, b, k-self.f[p]]
            self.R[p, b, k] = self.R[p - 1, b, k-self.f[p]]
        else:
            self.M[p, b, k] = float(np.inf)

        for q in range(p - 1, -1, -1):
            e = self.M[q, b - 1, k] + self.sqerror[q + 1, p]
            if e < self.M[p, b, k]:
                self.L[p, b, k] = q + 1
                self.R[p, b, k] = p
                self.M[p, b, k] = e


    def getBucketRanges(self):
        # Helper function that calculates the indexes of the values
        # that represent the edges of each bucket by using self.L, self.R

        # returns: left_edges = indices of the elements in self.values that are the left edges of the bukets (ascending order)
        #          right_edges = indices of the elements in self.values that are the right edges of the bukets (ascending order)
        #          n_elem = number of elements contained in the respective bucket
        # Only works with the method 'con'

        right_edges = np.zeros(self.beta, dtype=int)
        left_edges = np.zeros(self.beta, dtype=int)
        right_edges[self.beta - 1] = self.R[len(self.f)-1, self.beta-1, self.k]
        left_edges[self.beta - 1] = self.L[len(self.f)-1, self.beta-1, self.k]
        n = left_edges[self.beta - 1] - 1
        n = n.astype(int)
        n_el_no_del = self.sumf[len(self.f)-1]- self.sumf[left_edges[self.beta-1]] + self.f[left_edges[self.beta-1]]
        n_el_del = self.sumf[right_edges[self.beta-1]]- self.sumf[left_edges[self.beta-1]] + self.f[left_edges[self.beta-1]]
        test_del = n_el_no_del - n_el_del
        k = self.k - test_del
        k = k.astype(int)
        for b in range(self.beta - 2, -1, -1):
            left_edges[b] = self.L[n, b, k]
            right_edges[b] = self.R[n, b, k]
            if b == 0 :
                n_el_no_del = self.sumf[left_edges[b+1] - 1]- self.sumf[0] + self.f[0]
            else:
                n_el_no_del = self.sumf[left_edges[b+1] - 1]- self.sumf[left_edges[b]] + self.f[left_edges[b]]
            n_el_del = self.sumf[right_edges[b]]- self.sumf[left_edges[b]] + self.f[left_edges[b]]
            test_del = n_el_no_del - n_el_del
            k = k - test_del
            k = k.astype(int)
            n = left_edges[b] - 1
            n = n.astype(int)

        n_elem = np.zeros(self.beta)
        for i in range(self.beta):
            n_elem[i] = self.sumf[right_edges[i]] - self.sumf[left_edges[i]] + self.f[left_edges[i]]

        return left_edges, right_edges, n_elem

    def printSummary(self):
        if self.method != 'con':
            print("Method not supported when the method is not `con`")
            return
        left_edges, right_edges, n_elem = self.getBucketRanges()
        for i in range(self.beta):
            print(f'Bucket ({i+1}) {self.values[left_edges[i]], self.values[right_edges[i]]}:{n_elem[i]}')


    def preLowerMaxErr(self):
        # Utility function used to precompute the lower max error when needed
        for i in range(len(self.f)):
            for j in range(i, len(self.f)):
                for h in range(self.k + 1):
                    self.lme[i, j, h] = self.lowerMaxErr(i, j, h)
        return
