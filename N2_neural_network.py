# using python 3.9.6

# Standard library
import json
import random
import sys
import os

# Third-party libraries
import numpy as np
import pandas as pd

## define cost functions ##

class MeanSquaredErrorCost(object): # typical cost function for regression neural network

    @staticmethod
    def fn(a, y):
        cost = 0.5 * np.linalg.norm(a - y) ** 2
        return cost

    @staticmethod
    def delta(z, a, y):
        return (a - y) * relu_prime(z)
    
class MeanSquaredErrorCostWithConstraint(object): # cost function with constraint that prevents putting N atom at the origin

    @staticmethod
    def fn(a, y, r=1.0):
        # Original MSE cost
        cost = 0.5 * np.linalg.norm(a - y) ** 2
        
        # Calculate the distance from the origin for each coordinate set
        dist1 = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        dist2 = np.sqrt(a[3]**2 + a[4]**2 + a[5]**2)
        
        # Apply penalty if the predicted N atom is less than r from metal center (r = 1 Å by default)
        penalty = 0
        if dist1 < r:
            penalty += (r - dist1) ** 2
        if dist2 < r:
            penalty += (r - dist2) ** 2
        
        # Add the penalty to the cost
        cost += penalty
        
        return cost

    @staticmethod
    def delta(z, a, y, r=1.0):
        # Gradient of the original MSE cost
        delta = (a - y) * relu_prime(z)
        
        # Calculate the distance from the origin for each coordinate set
        dist1 = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        dist2 = np.sqrt(a[3]**2 + a[4]**2 + a[5]**2)
        
        # Apply penalty gradient if the predicted N atom is less than r from metal center (r = 1 Å by default)
        if dist1 < r:
            delta[0] += 2 * (r - dist1) * (-a[0] / dist1) * relu_prime(z[0])
            delta[1] += 2 * (r - dist1) * (-a[1] / dist1) * relu_prime(z[1])
            delta[2] += 2 * (r - dist1) * (-a[2] / dist1) * relu_prime(z[2])
        if dist2 < r:
            delta[3] += 2 * (r - dist2) * (-a[3] / dist2) * relu_prime(z[3])
            delta[4] += 2 * (r - dist2) * (-a[4] / dist2) * relu_prime(z[4])
            delta[5] += 2 * (r - dist2) * (-a[5] / dist2) * relu_prime(z[5])
        return delta

### main network class ##
class Network(object):

    def __init__(self, sizes, cost=MeanSquaredErrorCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    # initializer, distributes weights and biases over a gaussian distribution from 0 to 1
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    # initializer, applies exponential function that weights earlier entries more heavily
    def topheavy_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = []
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            weight_matrix = np.random.randn(y, x) / np.sqrt(x)
            decay_factors = np.exp(-np.arange(x) / x)
            weight_matrix *= decay_factors
            self.weights.append(weight_matrix)

    # feedforward algorithm, takes input vector, applies dot product for each network layer, returns output vector
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            if np.isnan(z).any():
                print("NaN detected, pre-activation, resetting...")
                return None
            a = relu(z)
            if np.isnan(a).any():
                print("NaN detected, activation, resetting...")
                return None
            if np.isinf(a).any():
                print("Infinity detected, activation, resetting...")
                return None
        return a
    
    # stochastic gradient decent algorithm, actually used to run the network and will call all the other functions
    def SGD(self, training_data, epochs, mini_batch_size, eta,
        lmbda=0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False,
        lr_decay=0.98):
    
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, mini_batch_size, eta)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, epochs, mini_batch_size, eta, lmbda, evaluation_data, lr_decay)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {}".format(accuracy))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, mini_batch_size, eta)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, epochs, mini_batch_size, eta, lmbda, evaluation_data, lr_decay)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {}".format(accuracy))

            # Decay the learning rate
            eta *= lr_decay

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
    
    # reshuffles subset of training examples to train on
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                    for b, nb in zip(self.biases, nabla_b)]

    #backpropagation algorithm, updates network weights and biases
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # determine distance between the predicted and actual N atom coordinate, in Å
    def accuracy(self, data, epochs, mini_batch_size, eta, lmbda, evaluation_data, lr_decay):
        results = [((self.feedforward(x)), y)
                   for (x, y) in data]
        return sum(dist_cart(x, y) for (x, y) in results) / len(results)

    # calculate total cost function for a given epoch
    def total_cost(self, data, lmbda, mini_batch_size, eta):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    # function to save network
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

## define auxiliary functions ##

# function to load network from a save file
def load_network(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

# leaky rectifier function and derivate
def relu(z):
    return np.maximum(0.01*z, z)

def relu_prime(z):
    return np.where(z > 0, 1.0, 0.01)

# measure Cartesian distance between two points
def dist_cart(predicted, actual):
    dist1 = np.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2 + (predicted[2] - actual[2]) ** 2)
    dist2 = np.sqrt((predicted[3] - actual[3]) ** 2 + (predicted[4] - actual[4]) ** 2 + (predicted[5] - actual[5]) ** 2)
    tot = (dist1 + dist2)/2 # divide by two to give the per-atom distance cost
    return tot

## load csv files and convert to 2D LnM and N2 vectors ##

def zero_fill(array, length):
    # Check if the array is shorter than the specified length
    if len(array) < length:
        # Calculate the number of zeros to add
        num_zeros_to_add = int(length - len(array))
        # Create an array of zeros with the required shape
        zeros_to_add = np.zeros(num_zeros_to_add)
        # Concatenate the original array with the zeros
        array = np.concatenate((array, zeros_to_add), axis=0)
    return array

# unwind csv file into 1D vector
def unwrap(file,length=100):
    os.chdir(os.path.abspath(file))
    file_name = os.path.basename(file)+"noN2.csv"
    no_N2table = pd.read_csv(file_name, header=None)
    x = []
    for row in no_N2table.iterrows():
        for j in range(4):
            x.append(row[1][j])
    if len(x) < length: #zero fill if vector smaller than 200 atoms
        x = zero_fill(x,length)
    else:             #shorten if vector larger than 200 atoms
        x = x[0:length]
    file_name = os.path.basename(file)+"onlyN2.csv"
    N2table = pd.read_csv(file_name, header=None)
    y = []
    for row in N2table.iterrows():
        for j in range(1,4):
            y.append(row[1][j])
    x_np = np.array(x,dtype=np.float64)
    x_np = x_np[:,np.newaxis]
    y_np = np.array(y,dtype=np.float64)
    y_np = y_np[:,np.newaxis]
    vector = (x_np,y_np)
    return vector

# compile unwrapped vectors into structure input file
def load_data(search_folder,length=100):
    count = os.listdir(os.path.abspath(search_folder))
    master_vector = []
    for i in count:
        vector = unwrap(os.path.abspath(search_folder)+'/'+i,length=length)
        master_vector.append(vector)
    return master_vector

# convert atomic number to group, period, and electronegativity
def AN_to_PC(training_data):
    updated_data = []
    for data in training_data:
        x, y = data
        new_x = []
        for i in range(len(x)):
            if i % 4 == 0:  # 0th, 4th, 8th, etc.
                coordinates = periodic_dictionary[int(x[i][0])]
                for n in coordinates:
                    new_x.append(np.array([np.array(n)]))
            else:
                new_x.append(np.array([x[i][0]]))
        temp = np.array(new_x)
        updated_data.append((temp, y))
    return updated_data

## define periodic dictionary ##
# key is the atomic number
# values are group, period, and electronegativity
# all values are normalized
periodic_dictionary = {
    0: [0, 0, 0],
    1: [1/18, 1/7, 0.55],
    2: [1, 1/7, 0],
    3: [1/18, 2/7, 0.245],
    4: [1/9, 2/7, 0.392],
    5: [13/18, 2/7, 0.51],
    6: [7/9, 2/7, 0.638],
    7: [15/18, 2/7, 0.76],
    8: [8/9, 2/7, 0.86],
    9: [17/18, 2/7, 0.995],
    10: [1, 2/7, 0],
    11: [1/18, 3/7, 0.232],
    12: [1/9, 3/7, 0.328],
    13: [13/18, 3/7, 0.403],
    14: [7/9, 3/7, 0.475],
    15: [15/9, 3/7, 0.548],
    16: [8/9, 3/7, 0.645],
    17: [17/18, 3/7, 0.79],
    18: [1, 3/7, 0],
    19: [1/18, 4/7, 0.205],
    20: [1/9, 4/7, 0.25],
    21: [1/6, 4/7, 0.34],
    22: [2/9, 4/7, 0.385],
    23: [5/18, 4/7, 0.408],
    24: [1/3, 4/7, 0.415],
    25: [7/18, 4/7, 0.388],
    26: [4/9, 4/7, 0.458],
    27: [1/2, 4/7, 0.47],
    28: [5/18, 4/7, 0.478],
    29: [11/18, 4/7, 0.475],
    30: [2/3, 4/7, 0.413],
    31: [13/18, 4/7, 0.453],
    32: [7/9, 4/7, 0.503],
    33: [15/18, 4/7, 0.545],
    34: [8/9, 4/7, 0.638],
    35: [17/18, 4/7, 0.74],
    36: [1, 4/7, 0],
    37: [1/18, 5/7, 0.205],
    38: [1/9, 5/7, 0.238],
    39: [1/6, 5/7, 0.305],
    40: [2/9, 5/7, 0.333],
    41: [5/18, 5/7, 0.4],
    42: [1/3, 5/7, 0.54],
    43: [7/18, 5/7, 0.475],
    44: [4/9, 5/7, 0.55],
    45: [1/2, 5/7, 0.57],
    46: [5/9, 5/7, 0.55],
    47: [11/18, 5/7, 0.483],
    48: [2/3, 5/7, 0.423],
    49: [13/18, 5/7, 0.445],
    50: [7/9, 5/7, 0.49],
    51: [15/18, 5/7, 0.513],
    52: [8/9, 5/7, 0.525],
    53: [17/18, 5/7, 0.665],
    54: [1, 5/7, 0.65],
    55: [1/18, 6/7, 0.198],
    56: [1/9, 6/7, 0.223],
    57: [1/6, 6/7, 0.275],
    58: [1/6, 6/7, 0.28],
    59: [1/6, 6/7, 0.283],
    60: [1/6, 6/7, 0.285],
    61: [1/6, 6/7, 0.29],
    62: [1/6, 6/7, 0.293],
    63: [1/6, 6/7, 0.295],
    64: [1/6, 6/7, 0.3],
    65: [1/6, 6/7, 0.303],
    66: [1/6, 6/7, 0.305],
    67: [1/6, 6/7, 0.3075],
    68: [1/6, 6/7, 0.31],
    69: [1/6, 6/7, 0.313],
    70: [1/6, 6/7, 0.315],
    71: [1/6, 6/7, 0.318],
    72: [2/9, 6/7, 0.325],
    73: [5/18, 6/7, 0.375],
    74: [1/3, 6/7, 0.59],
    75: [7/18, 6/7, 0.475],
    76: [4/9, 6/7, 0.55],
    77: [1/2, 6/7, 0.55],
    78: [5/9, 6/7, 0.57],
    79: [11/18, 6/7, 0.635],
    80: [2/3, 6/7, 0.5],
    81: [13/18, 6/7, 0.405],
    82: [7/9, 6/7, 0.583],
    83: [15/18, 6/7, 0.505],
    84: [8/9, 6/7, 0.5],
    85: [17/18, 6/7, 0.55],
    86: [1, 6/7, 0],
    87: [1/18, 1, 0.175],
    88: [1/9, 1, 0.225],
    89: [1/6, 1, 0.275],
    90: [1/6, 1, 0.325],
    91: [1/6, 1, 0.375],
    92: [1/6, 1, 0.345],
    93: [1/6, 1, 0.34],
    94: [1/6, 1, 0.32],
    95: [1/6, 1, 0.325],
    96: [1/6, 1, 0.325],
    97: [1/6, 1, 0.325],
    98: [1/6, 1, 0.325],
    99: [1/6, 1, 0.325],
    100: [1/6, 1, 0.325],
    101: [1/6, 1, 0.325],
    102: [1/6, 1, 0.325],
    103: [1/6, 1, 0.325]
}