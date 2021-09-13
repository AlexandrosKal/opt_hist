from summary import Summary
import numpy as np

class DeletionSummary(Summary):
    # Parent class of the summaries that allow deletions


    def __init__(self, values, f, beta, k):
        Summary.__init__(self, values, f, beta)
        self.k = k # number of deletions allowed

    def summarize(self):
        # Function that will be overriden by the children classes
        pass


    def arbSingleBucketError(self, p, q, k):

        # Algorithm that calculates the single bucket [p, q] error
        # with up to k arbitary deletions
        # Implemented according to Figure 1 in the paper
        # p, q = int, indices of the self.values and self.f arrays (frequency vector)
        # k = int, number of deletions allowed

        if p == q or np.sum(self.f[p:q+1]) <= k:
            return 0

        e_min = float(np.inf)
        left = p
        k_left = 0
        while k_left <= k:
            right = q
            k_right = 0
            while k_right <= k - k_left:
                k_mid = k - k_left - k_right
                e = self.lowerMaxErr(left, right, k_mid)
                e_min = np.fmin(e, e_min)
                k_right += self.f[right]
                right -= 1
            k_left += self.f[left]
            left += 1

        return e_min


    def lowerMaxErr(self, p, q, k):
        # Sub procedure used by arbSingleBucketError
        # p, q = int, indices of the self.values and self.f arrays (frequency vector)
        # k = int, number of deletions allowed

        if p == q :
            return 0
        first = p
        last = q
        f_copy = self.f[first:last+1].copy()

        p_min = f_copy.argmin()
        u_min = f_copy[p_min]
        p_max = f_copy.argmax()
        u_max = f_copy[p_max]
        while k > 0 and u_max > u_min:
            f_copy[p_max] -= 1
            k -= 1
            p_max = f_copy.argmax()
            u_max = f_copy[p_max]
        n_sum = np.sum(f_copy)
        n_sum_sq = np.sum(f_copy**2)
        return (n_sum_sq - n_sum**2 / (self.values[last] - self.values[first] + 1))


    def conSingleBucketError(self, p, q, k):
        # Algorithm that calculates the single bucket [p, q] error
        # with up to k consistent deletions
        # Implemented according to Figure 2 in the paper
        # p, q indices of the self.values and self.f arrays (frequency vector)
        # k number of deletions allowed

        if p == q :
            return 0, p, q, 0

        if np.sum(self.f[p:q+1]) <= k:
            return 0, -1, -1, k


        right = q
        k_right = 0
        while k_right + self.f[right] <= k:
                k_right += self.f[right]
                right -= 1

        left = p
        e_min = self.sqerror[left, right]
        k_left = 0
        best_l, best_r = left, right
        k_opt = k_left + k_right
        while k_left + self.f[left] <= k:
            k_left += self.f[left]
            left += 1
            while k_left + k_right > k:
                right += 1
                k_right -= self.f[right]
            sse = self.sqerror[left, right]
            if sse < e_min:
                best_l, best_r = left, right
                k_opt = k_left + k_right
            e_min = np.fmin(e_min, sse)
        return e_min, best_l, best_r, k_opt
