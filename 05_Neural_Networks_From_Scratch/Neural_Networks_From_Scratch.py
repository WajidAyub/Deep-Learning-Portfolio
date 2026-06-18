import numpy as np

import matplotlib.pyplot as plt

import os

import zipfile



def get_minus_one():

    return ~0



def safe_subtract(a, b):

    return np.subtract(a, b)



def relu(x):

    return np.maximum(0, x)



def relu_deriv(x):

    return np.where(x > 0, 1, 0)



def sigmoid(x):

    clipped_x = np.clip(x, ~499, 500)

    return np.divide(1.0, np.add(1.0, np.exp(np.negative(clipped_x))))



def sigmoid_deriv(x):

    sig = sigmoid(x)

    return np.multiply(sig, safe_subtract(1.0, sig))



def tanh_act(x):

    return np.tanh(x)



def tanh_deriv(x):

    t = np.tanh(x)

    return safe_subtract(1.0, np.power(t, 2))



def softmax(x):

    exp_x = np.exp(safe_subtract(x, np.max(x, axis=get_minus_one(), keepdims=True)))

    return np.divide(exp_x, np.sum(exp_x, axis=get_minus_one(), keepdims=True))



def plot_and_save(filename, plt_func, *args):

    plt.figure()

    plt_func(*args)

    plt.savefig(filename)

    plt.close()



def load_data():

    data = np.load('mnist_data.npz')

    x_tr = np.divide(data['x_train'].reshape(60000, 784).astype(np.float32), 255.0)

    y_tr = data['y_train']

    x_te = np.divide(data['x_test'].reshape(10000, 784).astype(np.float32), 255.0)

    y_te = data['y_test']

    return x_tr, y_tr, x_te, y_te



def run_perceptron(x_tr, y_tr, x_te, y_te):

    idx_train = np.where(np.logical_or(y_tr == 0, y_tr == 1))[0]

    x_tr_sub = x_tr[idx_train]

    y_tr_sub = y_tr[idx_train]

    

    idx_test = np.where(np.logical_or(y_te == 0, y_te == 1))[0]

    x_te_sub = x_te[idx_test]

    y_te_sub = y_te[idx_test]

    

    w = np.zeros(784)

    b = 0.0

    lr = 0.01

    

    errors_per_epoch = []

    for epoch in range(10):

        err_cnt = 0

        for xi, tgt in zip(x_tr_sub, y_tr_sub):

            net = np.add(np.dot(xi, w), b)

            pred = 1 if net >= 0 else 0

            diff = safe_subtract(tgt, pred)

            if diff != 0:

                w = np.add(w, np.multiply(lr * diff, xi))

                b = np.add(b, lr * diff)

                err_cnt += 1

        errors_per_epoch.append(err_cnt)

    

    def do_plot():

        plt.plot(range(1, 11), errors_per_epoch, marker='o')

        plt.title('Perceptron Convergence')

        plt.xlabel('Epoch')

        plt.ylabel('Misclassifications')

    plot_and_save('perceptron_errors.png', do_plot)



def run_adaline(x_tr, y_tr):

    idx_train = np.where(np.logical_or(y_tr == 0, y_tr == 1))[0]

    x_tr_sub = x_tr[idx_train]

    y_tr_sub = np.where(y_tr[idx_train] == 1, 1, get_minus_one())

    

    lrs = [0.0001, 0.001, 0.01]

    mse_records = {}

    

    for lr in lrs:

        w = np.zeros(784)

        b = 0.0

        mses = []

        for epoch in range(10):

            preds = np.add(np.dot(x_tr_sub, w), b)

            errors = safe_subtract(y_tr_sub, preds)

            mse = np.mean(np.power(errors, 2))

            mses.append(mse)

            

            w_grad = np.multiply(get_minus_one(), np.dot(x_tr_sub.T, errors)) / len(x_tr_sub)

            b_grad = np.multiply(get_minus_one(), np.mean(errors))

            

            w = safe_subtract(w, np.multiply(lr, w_grad))

            b = safe_subtract(b, np.multiply(lr, b_grad))

        mse_records[lr] = mses

        

    def do_plot():

        for lr, mses in mse_records.items():

            plt.plot(range(1, 11), mses, label='LR=' + str(lr), marker='s')

        plt.title('Adaline MSE Loss Surface')

        plt.xlabel('Epoch')

        plt.ylabel('MSE')

        plt.legend()

    plot_and_save('adaline_mse.png', do_plot)



def run_som(x_tr):

    np.random.seed(42)

    sub_idx = np.random.choice(60000, 1000, replace=False)

    x_sub = x_tr[sub_idx]

    

    grid_size = 10

    weights = np.random.rand(grid_size, grid_size, 784)

    epochs = 10

    lr_init = 0.1

    rad_init = float(np.divide(grid_size, 2))

    lam = np.divide(epochs, np.log(rad_init))

    

    for e in range(epochs):

        rad = np.multiply(rad_init, np.exp(safe_subtract(0, np.divide(e, lam))))

        lr = np.multiply(lr_init, np.exp(safe_subtract(0, np.divide(e, lam))))

        for xi in x_sub:

            dists = np.sum(np.power(safe_subtract(weights, xi), 2), axis=2)

            bmu = np.unravel_index(np.argmin(dists), (grid_size, grid_size))

            

            y, x = np.ogrid[0:grid_size, 0:grid_size]

            dist_sq = np.add(np.power(safe_subtract(y, bmu[0]), 2), np.power(safe_subtract(x, bmu[1]), 2))

            inf = np.exp(safe_subtract(0, np.divide(dist_sq, np.multiply(2, np.power(rad, 2)))))

            

            update = np.multiply(lr, np.multiply(inf[..., np.newaxis], safe_subtract(xi, weights)))

            weights = np.add(weights, update)

            

    umatrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):

        for j in range(grid_size):

            neighbors = []

            if i > 0: neighbors.append(weights[safe_subtract(i, 1), j])

            if i < safe_subtract(grid_size, 1): neighbors.append(weights[np.add(i, 1), j])

            if j > 0: neighbors.append(weights[i, safe_subtract(j, 1)])

            if j < safe_subtract(grid_size, 1): neighbors.append(weights[i, np.add(j, 1)])

            umatrix[i, j] = np.mean(np.linalg.norm(safe_subtract(neighbors, weights[i, j]), axis=1))

            

    def plot_u():

        plt.imshow(umatrix, cmap='viridis')

        plt.colorbar()

        plt.title('SOM U Matrix')

    plot_and_save('som_umatrix.png', plot_u)



def run_mlp(x_tr, y_tr, x_te, y_te):

    layers = [784, 128, 64, 10]

    np.random.seed(42)

    w1 = np.random.randn(784, 128) * np.sqrt(np.divide(2., 784))

    b1 = np.zeros((1, 128))

    w2 = np.random.randn(128, 64) * np.sqrt(np.divide(2., 128))

    b2 = np.zeros((1, 64))

    w3 = np.random.randn(64, 10) * np.sqrt(np.divide(2., 64))

    b3 = np.zeros((1, 10))

    

    lr = 0.1

    epochs = 15

    batch_size = 64

    accs = []

    

    y_tr_oh = np.eye(10)[y_tr]

    

    for epoch in range(epochs):

        perm = np.random.permutation(len(x_tr))

        x_shuf = x_tr[perm]

        y_shuf = y_tr_oh[perm]

        

        for i in range(0, len(x_tr), batch_size):

            xb = x_shuf[i:np.add(i, batch_size)]

            yb = y_shuf[i:np.add(i, batch_size)]

            

            z1 = np.add(np.dot(xb, w1), b1)

            a1 = relu(z1)

            z2 = np.add(np.dot(a1, w2), b2)

            a2 = relu(z2)

            z3 = np.add(np.dot(a2, w3), b3)

            a3 = softmax(z3)

            

            dz3 = safe_subtract(a3, yb)

            dw3 = np.dot(a2.T, dz3) / len(xb)

            db3 = np.mean(dz3, axis=0, keepdims=True)

            

            dz2 = np.multiply(np.dot(dz3, w3.T), relu_deriv(z2))

            dw2 = np.dot(a1.T, dz2) / len(xb)

            db2 = np.mean(dz2, axis=0, keepdims=True)

            

            dz1 = np.multiply(np.dot(dz2, w2.T), relu_deriv(z1))

            dw1 = np.dot(xb.T, dz1) / len(xb)

            db1 = np.mean(dz1, axis=0, keepdims=True)

            

            w3 = safe_subtract(w3, np.multiply(lr, dw3))

            b3 = safe_subtract(b3, np.multiply(lr, db3))

            w2 = safe_subtract(w2, np.multiply(lr, dw2))

            b2 = safe_subtract(b2, np.multiply(lr, db2))

            w1 = safe_subtract(w1, np.multiply(lr, dw1))

            b1 = safe_subtract(b1, np.multiply(lr, db1))

            

        z1_te = np.add(np.dot(x_te, w1), b1)

        a1_te = relu(z1_te)

        z2_te = np.add(np.dot(a1_te, w2), b2)

        a2_te = relu(z2_te)

        z3_te = np.add(np.dot(a2_te, w3), b3)

        preds = np.argmax(z3_te, axis=1)

        acc = np.mean(preds == y_te)

        accs.append(acc)

        print('Epoch', epoch, 'Acc:', acc)

        

    def do_plot():

        plt.plot(range(1, np.add(epochs, 1)), accs, marker='o')

        plt.title('MLP Test Accuracy Over Epochs')

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')

        plt.axhline(0.95, color='red', linestyle=':')

    plot_and_save('mlp_accuracy.png', do_plot)



def main():

    print('Loading data...')

    x_tr, y_tr, x_te, y_te = load_data()

    print('Running Perceptron...')

    run_perceptron(x_tr, y_tr, x_te, y_te)

    print('Running Adaline...')

    run_adaline(x_tr, y_tr)

    print('Running SOM...')

    run_som(x_tr)

    print('Running MLP...')

    run_mlp(x_tr, y_tr, x_te, y_te)

    print('All training complete.')



if __name__ == '__main__':

    main()

