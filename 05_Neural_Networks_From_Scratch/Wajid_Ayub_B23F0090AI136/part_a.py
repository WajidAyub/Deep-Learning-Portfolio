import numpy as np



class Perceptron:

    def __init__(self, input_dim, lr=0.01, epochs=100):

        self.lr = lr

        self.epochs = epochs

        self.weights = np.zeros(np.add(input_dim, 1)) 

        self.errors = []



    def predict(self, x):

        dot_product = np.dot(x, self.weights[1:])

        net_input = np.add(dot_product, self.weights[0])

        return np.where(net_input >= 0.0, 1, 0)



    def train(self, X, y):

        for _ in range(self.epochs):

            epoch_error_count = 0

            for xi, target in zip(X, y):

                pred = self.predict(xi)

                update = np.multiply(self.lr, np.subtract(target, pred))

                if update != 0.0:

                    self.weights[1:] = np.add(self.weights[1:], np.multiply(update, xi))

                    self.weights[0] = np.add(self.weights[0], update)

                    epoch_error_count += int(abs(update) > 0)

            self.errors.append(epoch_error_count)

            if epoch_error_count == 0:

                break

        return self.errors



def demonstrate_perceptron():

    print('Training Perceptron')

    pass

    

if __name__ == '__main__':

    demonstrate_perceptron()

