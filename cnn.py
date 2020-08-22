from load_data import train_data, test_data
import pywt
import scipy.misc
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import block_reduce
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def plot_wavelet(time, signal, scales, waveletname='cmor', cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period', xlabel='Time'):
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    print(coefficients.shape)
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period + 1e-10), np.log2(power + 1e-10), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


def get_coefficients(signal, scales, waveletname='cmor'):
    [coefficients, _] = pywt.cwt(signal, scales, waveletname)
    power = (abs(coefficients)) ** 2
    # power = np.log2(power + 1e-10)
    return power


def train_coefficients_array(n1=0, n2=train_data.shape[0]):
    sc = np.arange(1, 128)
    outputs_all = None
    for i in range(n1, n2):
        sig = train_data.iloc[[i]].values.tolist()[0][:-1]
        output = block_reduce(get_coefficients(sig, sc), block_size=(3, 4), func=np.mean)
        if i == n1:
            outputs_all = np.expand_dims(output, axis=0)
        else:
            outputs_all = np.r_[outputs_all, np.expand_dims(output, axis=0)]
        if (i - n1) > 0 and (i - n1) % 1000 == 0:
            print("Processed: %d images" % (i - n1))
    return outputs_all


def test_coefficients_array(n1=0, n2=test_data.shape[0]):
    sc = np.arange(1, 128)
    outputs_all = None
    for i in range(n1, n2):
        sig = test_data.iloc[[i]].values.tolist()[0][:-1]
        output = block_reduce(get_coefficients(sig, sc), block_size=(3, 4), func=np.mean)
        if i == n1:
            outputs_all = np.expand_dims(output, axis=0)
        else:
            outputs_all = np.r_[outputs_all, np.expand_dims(output, axis=0)]
        if (i - n1) > 0 and (i - n1) % 1000 == 0:
            print("Processed: %d images" % (i - n1))
    return outputs_all


def define_model(array):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(array.shape[0], array.shape[1], 1)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    all_test_images = test_coefficients_array()
    np.save('data/test_images', all_test_images)
# all_train_images = coefficients_array()
# np.save('data/train_images', all_train_images)

# array1 = coefficients_array(40000, 40001)
# array1 = array1[0]
# plt.imshow(np.asarray(array1), cmap='jet')
# plt.imshow(np.asarray(array2))
# plt.show()
# define_model(array1)
# print(array1)
