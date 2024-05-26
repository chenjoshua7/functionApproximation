import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from utils import *
from sklearn.preprocessing import StandardScaler

def relu(x):
    return np.maximum(0, x)

def cost(ypred, y):
    return np.sum((ypred.flatten() - y.flatten())**2)/len(y.flatten())

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class FCNN:
    def __init__(self, layer_dims) -> None:
        self.params = {}
        self.params["layers"] = len(layer_dims) - 1
        for l in range(1,len(layer_dims)):
            self.params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            self.params["b" + str(l)] = np.zeros((layer_dims[l],1))
        self.cache = None
        
    def forward(self, X):
        cache = {}
        
        layers = self.params["layers"]
        A = X.T
        
        for l in range(1, layers + 1):
            Wl = self.params["W" + str(l)]
            bl = self.params["b" + str(l)]
            
            Zl = np.dot(Wl, A) + bl
            if l == layers:
                Al = Zl
            else:
                Al = relu(Zl)
            
            cache["Z" + str(l)] = Zl
            cache["A" + str(l)] = Al
            A = Al
        
        self.cache = cache
        return Al, cache
        
    def backward(self, y, xs):
        layers = self.params["layers"]
        grads = {}
        
        Al = self.cache["A" + str(layers)]  # (1 x 1000)
        dAl = -2 * (y - Al)
        dZl = dAl 
        
        grads["dW" + str(layers)] = np.dot(dZl, self.cache["A" + str(layers - 1)].T) / y.shape[0] # (1, 10)
        grads["db" + str(layers)] = np.mean(dZl, axis=1, keepdims=True)
        
        for l in range(layers - 1, 0, -1):
            dAl = np.dot(self.params["W" + str(l + 1)].T, dZl)
            dZl = dAl * relu_derivative(self.cache["Z" + str(l)])
            
            grads["dW" + str(l)] = np.dot(dZl, self.cache["A" + str(l - 1)].T if l != 1 else xs) / y.shape[0]
            grads["db" + str(l)] = np.mean(dZl, axis=1, keepdims=True)
        
        self.grads = grads


    def teach_weights(self, lr, beta=0.9):
        layers = self.params["layers"]
        for l in range(1, layers + 1):
            if "Vdw" + str(l) not in self.params:
                self.params["Vdw" + str(l)] = np.zeros_like(self.params["W" + str(l)])
                self.params["Vdb" + str(l)] = np.zeros_like(self.params["b" + str(l)])
            
            # Update momentum
            self.params["Vdw" + str(l)] = beta * self.params["Vdw" + str(l)] + (1 - beta) * self.grads["dW" + str(l)]
            self.params["Vdb" + str(l)] = beta * self.params["Vdb" + str(l)] + (1 - beta) * self.grads["db" + str(l)]
            
            # Update weights and biases
            self.params["W" + str(l)] -= lr * self.params["Vdw" + str(l)]
            self.params["b" + str(l)] -= lr * self.params["Vdb" + str(l)]
            
class FCNN_approx:
    def __init__(self) -> None:
        self.graph_params = {}
        self.params = {}
        self.xs = None
        self.ys = None
        self.model = None
        
    def set_params(self, N=5, A=0):
        self.params["N"] = N
        self.params["A"] = A
        return
    
    def set_graph_params(self, size=5000, lower_bound=-10, upper_bound=10):
        self.graph_params = {
            "size": size,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def generate_data(self, func="exponential"):
        if func == "exponential":
            function = np.exp
        elif func == "cosine":
            function = np.cos
        elif func == "sine":
            function = np.sin
        else:
            function = func
            
        size = self.graph_params["size"]
        lower_bound = self.graph_params["lower_bound"]
        upper_bound = self.graph_params["upper_bound"]
        
        xs = np.sort(np.random.uniform(low=lower_bound, high=upper_bound, size=size)) 
        self.xs = create_polynomials(xs, N=self.params["N"])
        self.ys = function(xs)
        
    def initialize(self, N, func):
        self.set_params(N=N)
        self.set_graph_params()
        self.generate_data(func=func)
        
    def train(self, layer_dims, initial_lr, epochs=10, decay_rate=0.8, decay_steps=10, standardize=True, visualize=True, delay = 0.01):
        assert layer_dims[0] == self.params["N"], f'First dimension must have {self.params["N"]} inputs'
        assert layer_dims[-1] == 1, f'Last dimension must have output 1'
        
        if standardize:
            scaler = StandardScaler()
            xs = scaler.fit_transform(self.xs)
        else:
            xs = self.xs
        
        if self.model is None:
            self.model = FCNN(layer_dims=layer_dims)
        
        progress_bar = range(1, epochs + 1)
        for e in progress_bar:
            Al, cache = self.model.forward(xs)

            current_cost = cost(self.ys, Al)
            #progress_bar.set_postfix({'train_loss': f'{current_cost:.4f}'})
            if e % 25 == 0:
                if visualize:
                    clear_output(wait=True)
                    
                    self.plot_function(y_pred=Al, epoch = e, loss = round(current_cost,4))
                    display(plt.gcf())
                    time.sleep(delay) 
        
            self.model.backward(self.ys, self.xs)
            lr = initial_lr * (decay_rate ** (e / decay_steps))
            params = self.model.teach_weights(lr=lr) 

            
        return Al, params
    
    def plot_function(self, y_pred, **kwargs):
        lower_bound = self.graph_params["lower_bound"]
        upper_bound = self.graph_params["upper_bound"]

        title_parts = [f"{key.capitalize()} : {value}" for key, value in kwargs.items()]
        title = "NN approximation \n" + " - ".join(title_parts) if title_parts else "NN approximation"
        
        plt.plot(self.xs[:,0].flatten(), self.ys.flatten(), label='True', alpha=0.6, linewidth=4)
        plt.plot(self.xs[:,0].flatten(), y_pred.flatten(), label='Predicted', alpha=0.6, linewidth=4, color = "red")
        
        #plt.scatter(self.xs[:,0], y_pred, label='Predicted', color = "red", alpha=0.6)
        #plt.scatter(self.xs[:,0], self.ys, label='True', alpha=0.6)
        
        plt.title(title)
        
        plt.xlim(lower_bound * 1.1, upper_bound * 1.1)
        plt.ylim(np.min(self.ys) - abs(0.1 * np.max(self.ys)), np.max(self.ys) * 1.1)
        plt.legend()
        plt.show()