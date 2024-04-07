import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """Example: [2, 3, 1] === 3 layers with 2, 3 and 1 neurons respectively
        (First layer is the input layer)"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # Random initialization => gives a place for the stochastic gradient descent algorithm to start from
        # Network.weights[1] is a numpy matrix storing the weights connecting the second and third layers of neurons
        #  <=> matrix W
        #   => Wjk is the weight for the connection between the Kth neuron in the second layer
        #      and the Jth neuron in the third layer
        #   => vector of activations of the third layer of neurons is:
        #      a' = o(W a + b) --- a
        #   so  a' = sigmoid (to get range [0, 1]) of: the product of W
        #            (weighted connections) with the activation vector of the previous layer
        #            plus the bias; (W and b are matrices initiated randomly)

    def feedforward(self, a):  # 'a' is an (n, 1) Numpy ndarray
        """Apply equation explained above for each layer"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent
        The training data is a list of tuples (x, y) representing the training inputs and the desired outputs
        If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        Eta is the learning rate"""

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)  # shuffle training data
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]  # partition the shuffled training data into mini batches

            for mini_batch in mini_batches:
                # updates the network weights and biases according to a single iteration of gradient descent
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Uses backpropagation"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # creates a new NumPy array filled with zeros
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # with the same shape as the array w / b
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # uses the chain rule
        # cost derivative = (aL - desired output y) * o'(z(L))
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta  # -- for the bias
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # a(L - 1)    -- for the weights

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def cost_derivative(output_activations, y):  # ((aL - y)^2)'
    return 2 * (output_activations - y)  # I added a (* 2) because I think that's the correct result of the derivative


def sigmoid(z):  # o(z)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # o'(z)
    return sigmoid(z) * (1 - sigmoid(z))
