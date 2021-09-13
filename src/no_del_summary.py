from summary import Summary
import numpy as np

class NoDeletionSummary(Summary):

    def __init__(self, values, f, beta):
        Summary.__init__(self, values, f, beta)
        self.M = np.zeros((len(f), beta)) # Matrix that stores the error values. M[q, b] = optimal error until index q with b buckets
        self.L = np.zeros((len(f), beta)) # Matrix that holds the index of the value that is the left edge of the rightmost bucket

    def summarize(self):

        # Find the optimal summary of the dataset with beta buckets without deletions.
        # Implemented according to Figure 3 in the paper.
        # returns M[len(self.f)-1][self.beta-1] that is the minimum error when
        # we cover the whole dataset with beta buckets.

        for p in range(len(self.f)):
            self.M[p, 0] = self.sqerror[0, p]

        minl = 0
        for b_prime in range(1, self.beta):
            for q in range(1, len(self.f)):
                self.M[q, b_prime] = float(np.inf)
                for p in range(q-1, -1, -1):
                    e = self.M[p, b_prime-1] + self.sqerror[p + 1][q]
                    if e < self.M[q, b_prime]:
                        self.M[q, b_prime] = e
                        minl = p + 1

                self.L[q, b_prime] = minl # store the leftmost edge of rightmost bucket

        return self.M[len(self.f)-1][self.beta-1]

    def getBucketRanges(self):
        # Helper function that calculates the indexes of the values
        # that represent the edges of each bucket by using self.L

        # returns: left_edges = indices of the elements in self.values that are the left edges of the bukets (ascending order)
        #          right_edges = indices of the elements in self.values that are the right edges of the bukets (ascending order)
        #          n_elem = number of elements contained in the respective bucket


        right_edges = np.zeros(self.beta, dtype=int)
        left_edges = np.zeros(self.beta, dtype=int)

        # edges of the last bucket
        right_edges[self.beta - 1] = len(self.f) - 1
        left_edges[self.beta - 1] = self.L[len(self.f)-1, self.beta-1]
        n = left_edges[self.beta - 1] - 1

        # iteratively find the edges of the buckets from the second to last
        # until the first bucket
        for b in range(self.beta - 2, -1, -1):
            left_edges[b] = self.L[n, b]
            right_edges[b] = n
            n = left_edges[b]  - 1

        # Given the bucket edges found previously calculate the number
        # of elements inside each bucket
        n_elem = np.zeros(self.beta)
        for i in range(self.beta):
            n_elem[i] = self.sumf[right_edges[i]] - self.sumf[left_edges[i]] + self.f[left_edges[i]]

        return left_edges, right_edges, n_elem

    def printSummary(self):
        # Print a text representation of  the Summary following the format:
        # Bucket (number) [bucket_start, bucket_end]: number of elements inside the bucket')

        left_edges, right_edges, n_elem = self.getBucketRanges()
        for i in range(self.beta):
            print(f'Bucket ({i+1}) {self.values[left_edges[i]], self.values[right_edges[i]]}:{n_elem[i]}')
