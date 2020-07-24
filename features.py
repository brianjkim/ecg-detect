import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

train_data = pd.read_csv('data/train.csv')
train_data.rename(columns={'0.000000000000000000e+00.88': 'labels'}, inplace=True)
train_data.drop('labels', axis=1, inplace=True)


# convert dataframe to numpy array
train_data = train_data.values

# Column 0: mean of data points
features = np.mean(train_data, axis=1)

# alternate solution of combining numpy arrays: features = np.vstack([features, np.std(train_data, axis=1)]).T

# Column 1: standard deviation of data points
features = np.c_[features, np.std(train_data, axis=1)]

# Column 2: skewness of data points
features = np.c_[features, skew(train_data, axis=1)]

# Column 3: kurtosis of data points
features = np.c_[features, kurtosis(train_data, axis=1)]

# find number of points for each sample in a given 2d array where the first derivative equals zero. Returns a 1d array.


def zero_crossings(array):
    arr = []
    for lst in array:
        count = 0
        for i in range(0, len(lst) - 1):
            if lst[i] * lst[i + 1] < 0:
                count += 1
        arr.append([count])
    return arr


# creates an array where each point is the difference between adjacent points across axis 1 of train_data
diff = np.diff(train_data, axis=1)

# Column 4: number of points such that first derivative equals zero
features = np.c_[features, zero_crossings(diff)]

# creates an array that has elements that are the absolute value of those in diff
abs_diff = np.abs(diff)

# Column 5: sum of absolute value of data points in diff
features = np.c_[features, np.sum(abs_diff, axis=1)]
print(features[:5][:])


