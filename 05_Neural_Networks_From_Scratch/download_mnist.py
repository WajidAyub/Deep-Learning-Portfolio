import torchvision

import numpy as np

import os



def download_data():

    train_data = torchvision.datasets.MNIST(root='.', train=True, download=True)

    test_data = torchvision.datasets.MNIST(root='.', train=False, download=True)

    

    x_train = train_data.data.numpy()

    y_train = train_data.targets.numpy()

    x_test = test_data.data.numpy()

    y_test = test_data.targets.numpy()

    

    np.savez('mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    print('successfully downloaded and saved.')



if __name__ == '__main__':

    download_data()

