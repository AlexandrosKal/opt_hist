from del_summary import DeletionSummary
from no_del_summary import NoDeletionSummary
import numpy as np

class SDSummary(DeletionSummary):
    # Implementation of the summarize and then delete methods described in the paper.
    def __init__(self, values, f, beta, k, method='con'):
        DeletionSummary.__init__(self, values, f, beta, k)

        self.L = np.zeros((beta, k + 1)) # Matrix used to store the left edge of the beta bucket with k deletions
        self.R = np.zeros((beta, k + 1)) # Matrix used to store the right edge of the beta bucket with k deletions

        self.M = np.zeros((beta, k + 1)) # Matrix that stores the error values. M[b, k] = optimal error with b buckets and k deletions


        self.method = method # used to specify which method to use
        # 'con' for consistent deletions
        # 'arb' for arbitary deletions

    def summarize(self, k):
        # Find the optimal summary that allows for k deletions
        # using the summarize then delete strategy.
        # Start with the optimal buckets with no deletions.
        # Apply Consistent or Arbitary deletions on each bucket to minimize the error.
        # Implemented according to Figure 4 in the paper.
        # Depending on self.method applies the method selected
        # k = int, number of deletions allowed
        # returns M[self.beta-1][k] that is the minimum error with
        # beta buckets and k deletions.

        # Get the buckets with NoDelSummary and then remove elements
        nd_summary = NoDeletionSummary(self.values, self.f, self.beta)
        err = nd_summary.summarize()
        left_edges, right_edges, n_elem = nd_summary.getBucketRanges()


        p1 = left_edges[0]
        q1 = right_edges[0]
        e_min = float('inf')
        for k_prime in range(k + 1):
            if self.method == 'con':
                self.M[0, k_prime], opt_r , opt_l, _ = self.conSingleBucketError(p1, q1, k_prime)
                if self.M[0, k_prime] < e_min:
                    self.L[0, k_prime] = opt_r # left edge of rightmost bucket
                    self.R[0, k_prime] = opt_l # right edge of rightmost bucket
                    e_min = self.M[0, k_prime]
            elif self.method == 'arb':
                self.M[0, k_prime] = self.arbSingleBucketError(p1, q1, k_prime)
            else:
                print("Wrong method provided.")
                return -1

        B = (left_edges, right_edges, n_elem)
        for b_prime in range(1, self.beta):
            for k_prime in range(k + 1):
                self.SDUpdateMatrix(b_prime, k_prime, B)
        return self.M[self.beta - 1, k]

    def SDUpdateMatrix(self, b, k, B):
        # Sub procedure used by self.summarize to update the error matrix M.
        # Implemented according to Figure 4 of the paper.
        # p = int, index of the self.values and self.f arrays (frequency vector)
        # B = tuple of lists of integers: [left_edges, right_edges, n_elem]
        # k = int, number of deletions

        self.M[b, k] = float(np.inf)
        ps, qs, n_el = B
        p = ps[b]
        q = qs[b]
        n = n_el[b]
        opt_l = 0
        opt_r = 0
        minl = p
        minr = q
        for k_prime in range(k + 1):
            e = self.M[b-1, k-k_prime]
            if self.method == 'con':
                e_prime, opt_l, opt_r, _ = self.conSingleBucketError(p, q, k_prime)
            elif self.method == 'arb':
                e_prime = self.arbSingleBucketError(p, q, k_prime)
            else:
                print("Wrong method provided.")
                return -1
            if e + e_prime < self.M[b, k]:
                minl = opt_l
                minr = opt_r
            self.M[b, k] = np.fmin(self.M[b, k], e + e_prime)
        self.L[b, k] = minl # left edge of rightmost bucket
        self.R[b, k] = minr # right edge of rightmost bucket

    def getBucketRanges(self):
        # Helper function that calculates the indexes of the values
        # that represent the edges of each bucket by using self.L

        # returns: left_edges = indices of the elements in self.values that are the left edges of the bukets (ascending order)
        #          right_edges = indices of the elements in self.values that are the right edges of the bukets (ascending order)
        #          n_elem = number of elements contained in the respective bucket
        # Only available for the method 'con'

        if self.method != 'con':
            print("Method not supported when the method is not `con`")
            return [], [], []

        right_edges = np.zeros(self.beta, dtype=int)
        left_edges = np.zeros(self.beta, dtype=int)
        right_edges[self.beta - 1] = self.R[self.beta-1, self.k]
        left_edges[self.beta - 1] = self.L[self.beta-1, self.k]
        n_el_no_del = self.sumf[len(self.f)-1]- self.sumf[left_edges[self.beta-1]] + self.f[left_edges[self.beta-1]]
        n_el_del = self.sumf[right_edges[self.beta-1]]- self.sumf[left_edges[self.beta-1]] + self.f[left_edges[self.beta-1]]

        test_del = n_el_no_del - n_el_del
        k = self.k - test_del
        k = k.astype(int)

        for b in range(self.beta - 2, -1, -1):
            left_edges[b] = self.L[b, k]
            right_edges[b] = self.R[b, k]
            if b == 0 :
                n_el_no_del = self.sumf[left_edges[b+1] - 1]- self.sumf[0] + self.f[0]
            else:
                n_el_no_del = self.sumf[left_edges[b+1] - 1]- self.sumf[left_edges[b]] + self.f[left_edges[b]]
            n_el_del = self.sumf[right_edges[b]]- self.sumf[left_edges[b]] + self.f[left_edges[b]]
            test_del = n_el_no_del - n_el_del
            k = k - test_del
            k = k.astype(int)

        n_elem = np.zeros(self.beta)
        for i in range(self.beta):
            n_elem[i] = self.sumf[right_edges[i]] - self.sumf[left_edges[i]] + self.f[left_edges[i]]

        return left_edges, right_edges, n_elem


    def printSummary(self):
    # Print a text representation of  the Summary following the format:
    # Bucket (number) [bucket_start, bucket_end]: number of elements inside the bucket')
    # Only available for the method 'con'

        if self.method != 'con':
            print("Method not supported when the method is not `con`")
            return
        left_edges, right_edges, n_elem = self.getBucketRanges()
        for i in range(self.beta):
            print(f'Bucket ({i+1}) {self.values[left_edges[i]], self.values[right_edges[i]]}:{n_elem[i]}')
