from abc import ABC, abstractmethod
import numpy as np

class Summary(ABC):
    # Parent class for all the summaries

    def __init__(self, values, f, beta):

        # the pair of arrays (values, f) represent the frequency vector mentioned in the paper

        self.values = values # The unique values found in the dataset, this array must be sorted
        self.f = f # The frequency of each value, each element f[i] represents the frequency of values[i]

        self.beta = beta # The number of buckets to be used in the summary

        self.sumf = np.zeros(len(f)) # Matrix with the sum of the frequences up to each index
        self.sqsumf = np.zeros(len(f)) # Matrix with the sum of the frequences up to each index
        self.sumf[0] = f[0]
        self.sqsumf[0] = f[0]**2


        # Pre-compute the SSE as described in the paper

        for i in range(1, len(f)):
            self.sumf[i] = self.sumf[i - 1] + f[i]
            self.sqsumf[i] = self.sqsumf[i - 1] + f[i]**2

        self.sqerror = np.zeros((len(f), len(f)))
        for i in range(len(f)):
            for j in range(i, len(f)):
                self.sqerror[i, j] = self.calcSqError(i, j)

        self.N = len(values) # The number of unique values in the dataset


    def calcSqError(self, p, q):
        # Calculate the squared error from the index p till the index q of the frequency vector
        # p, q = int, indices of the self.values and self.f arrays (frequency vector)
        temp_sum = 0
        temp_sqsum = 0
        denom = self.values[q] - self.values[p] + 1
        if p != 0 :
            temp_sqsum = self.sqsumf[q] - self.sqsumf[p-1]
            temp_sum = self.sumf[q] - self.sumf[p-1]
        else:
            temp_sqsum = self.sqsumf[q]
            temp_sum = self.sumf[q]
        return (temp_sqsum - (temp_sum**2) / denom)


    def sumFreq(self, p, q):
        # Calculate the sum from the index p till the index q of the frequency vector
        # p, q = int, indices of the self.values and self.f arrays (frequency vector)
        if p != 0 :
            return self.sumf[q] - self.sumf[p-1]
        return self.sumf[q] - self.sumf[p] + self.f[p]


    @abstractmethod
    def summarize(self):
        pass
