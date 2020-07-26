import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('data/train.csv')
#Take first row of train_data and graph
# sample_data = train_data.iloc[[0]]
# sample_data = sample_data.values.tolist()[0]
# plt.plot(sample_data)
# plt.title('Sample ECG Data')
#
# plt.show()
train_data.rename(columns={'0.000000000000000000e+00.88': 'labels'}, inplace=True)
# label_zeros = train_data[train_data.labels == 0]
# num_rows_zeros = len(label_zeros.index)
#
#
# def random_row_zeros():
#     return label_zeros.iloc[np.random.randint(0, num_rows_zeros)].values.tolist()[0]
#
#
# label_ones = train_data[train_data.labels == 1]
# num_rows_ones = len(label_ones.index)
#
#
# def random_row_ones():
#     return label_ones.iloc[np.random.randint(0, num_rows_ones)].values.tolist()[0]
#
#
# label_twos = train_data[train_data.labels == 2]
# num_rows_twos = len(label_twos.index)
#
#
# def random_row_twos():
#     return label_twos.iloc[np.random.randint(0, num_rows_twos)].values.tolist()[0]
#
#
# label_threes = train_data[train_data.labels == 3]
# num_rows_threes = len(label_threes.index)
#
#
# def random_row_threes():
#     return label_threes.iloc[np.random.randint(0, num_rows_threes)].values.tolist()[0]
#
#
# label_fours = train_data[train_data.labels == 4]
# num_rows_fours = len(label_fours.index)
#
#
# def random_row_fours():
#     return label_fours.iloc[np.random.randint(0, num_rows_fours)].values.tolist()[0]


# Returns a list of lists that contain sample ECG recordings, with shape (hi - lo) x (num)
# Inputs
#   lo: lowest label index
#   hi: highest label index
#   num: number of times to repeat sample generation for each label
def random_rows(lo=0, hi=5, num=9):
    outputs = list()
    for ii in range(lo, hi):
        label_outputs = list()
        labels = train_data[train_data.labels == ii]
        num_rows = len(labels.index)
        for _ in range(num):
            random_output = labels.iloc[[np.random.randint(0, num_rows)]].values.tolist()[0][:-1]
            label_outputs.append(random_output)
        outputs.append(label_outputs)
    return outputs


def plot_samples(label):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(random_rows()[label][i])
    plt.show()


def get_labels():
    return train_data['labels'].values


# plot_samples(0)

