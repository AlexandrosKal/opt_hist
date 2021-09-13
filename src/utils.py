import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


def generate_norm_data(low, high, var=0.25, size=50_000):
    # Generate a random dataset that follows the normal distribution
    # Draws size samples from [low, high]
    # Returns: two arrays (values, f) that represent the frequency vector.

    data = np.random.normal((high-low)/2, var*((high-low)/2), size).astype(int)
    data = np.clip(data, low, high)
    values, f = np.unique(data, return_counts=True)
    return values, f

def rand_zipf(n, alpha, num_samples):
    # A random number generator following the zipf distribution
    # this is an implementation that allows for an exponent <=1.
    # Based on :
    # https://stackoverflow.com/questions/31027739/python-custom-zipf-number-generator-performing-poorly
    # Draws num_samples samples form [1, n]
    # alpha = the zeta exponent, in the paper we are studying they use alpha = 0.85

    # Calculate Zeta values from 1 to n:
    tmp = np.power( np.arange(1, n+1), -alpha )
    zeta = np.r_[0.0, np.cumsum(tmp)]
    # Store the translation map:
    distMap = [x / zeta[-1] for x in zeta]
    # Generate an array of uniform 0-1 pseudo-random values:
    u = np.random.random(num_samples)
    # bisect them with distMap
    v = np.searchsorted(distMap, u)
    samples = [t-1 for t in v]
    return samples


def generate_permzipf_data(low, high, alpha=0.85, size=50_000):
    # permute the frequencies drawn from the zipfian distribution
    # Draws size samples form [low, high]
    # alpha = the zeta exponent, in the paper we are studying they use alpha = 0.85
    # Returns: two arrays (values, f) that represent the frequency vector.

    data = rand_zipf(high, alpha, size)
    data = np.clip(data, low, high)
    values, f = np.unique(data, return_counts=True)
    f = np.random.permutation(f)

    return values, f


def plot_histogram(left, right, n):
    # Given the left and right values of each bucket edge and the elements inside each bucket
    # Create a visual representation of the summary
    # left = values that represent the left edges of the buckets (ascending order)
    # right = values that represent the right edges of the buckets (ascending order)
    # n = number of elements in each respective bucket
    buckets = []
    widths = []
    for i in range(len(left)):
        buckets.append(left[i]+right[i])
        width = right[i]-left[i]
        if width == 0:
            width = 0.1
        widths.append(width)
    plt.figure(figsize=(10,8))
    plt.title('Summary')
    plt.bar(np.array(buckets)/2, n, width=widths)

def to_freq_vec(data):
    # Transform the randomly generated data into
    # two arrays (values, f) that represent the frequency vector.
    values, f = np.unique(data, return_counts=True)
    # f = np.random.permutation(f)
    return values, f

def sample_data(values, f, frac):
    # Take uniform sample of the input dataset.
    # The sample is a percentage represented by frac.
    # frac = float between 0. and 1.
    # Return: two arrays (values, f) that represent the frequency vector of the sample.

    data = np.repeat(values, f)
    data_sampled = np.random.choice(data, size = int(len(data) * frac), replace=False)
    values, f = np.unique(data_sampled, return_counts=True)
    return values, f

def outlier_removal(values, f, cont=0.02):
    # Take uniform sample of the input dataset.
    # The sample is a percentage represented by frac.
    # frac = float between 0. and 1.
    # Return: two arrays (values, f) that represent the frequency vector of the sample.

    data = np.repeat(values, f)
    clf = LocalOutlierFactor(contamination=0.02)
    res = clf.fit_predict(data.reshape(-1, 1))
    print('outliers')
    print(len(data[res == -1]))
    data = data[res == 1]
    values, f = np.unique(data, return_counts=True)
    return values, f



