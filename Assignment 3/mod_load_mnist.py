import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

datasets_dir = './'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noValSamples=100, noTsSamples=100,\
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noValPerClass=10, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noValSamples==noValPerClass*len(digit_range), 'noValSamples and noValPerClass mismatch'
    
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    valX = np.zeros((noValSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)
    valY = np.zeros(noValSamples)

    count = 0
    for ll in digit_range:
        # Train and Validation data
        idl = np.where(trLabels == ll)
        PerClass = noTrPerClass + noValPerClass
        idl_tr = idl[0][: noTrPerClass]
        idl_val = idl[0][noTrPerClass: noTrPerClass+noValPerClass]

        idx_tr = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        idx_val = list(range(count*noValPerClass, (count+1)*noValPerClass))

        trX[idx_tr, :] = trData[idl_tr, :]
        trY[idx_tr] = trLabels[idl_tr]

        valX[idx_val, :] = trData[idl_val, :]
        valY[idx_val] = trLabels[idl_val]

        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    valX = valX.T

    valY = valY.reshape(1, -1)
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, valX, valY, tsX, tsY


def main():
    trX, trY, valX, valY, tsX, tsY = mnist(noTrSamples=30, noValSamples=15, noTsSamples=15, digit_range=[0, 5, 8], noTrPerClass=10, noValPerClass=5, noTsPerClass=5)

    
    plt.imshow(trX[:,5].reshape(28, -1))
    plt.show()
    trY[0,5]
    
if __name__ == "__main__":
    main()
