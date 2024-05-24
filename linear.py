import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from utils import *

class LinearApprox:
    def __init__(self) -> None:
        self.graph_params = {}
        self.params = {}
        self.xs = None
        self.coefs = None
    
        
    def set_graph_params(self, size = 100000, lower_bound = -10, upper_bound = 10):
        self.graph_params = {
            "size": size,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def plot_function(self, N =5, func="exponential"):    
        if func == "exponential":
            function = np.exp
        elif func == "cosine":
            function = np.cos
        elif func == "sine":
            function = np.sin
        else:
            function = func
            
        if not self.graph_params:
            self.set_graph_params()
        
        size = self.graph_params["size"]
        lower_bound = self.graph_params["lower_bound"]
        upper_bound = self.graph_params["upper_bound"]
        
        if self.xs is None:
            self.xs = np.sort(np.random.uniform(low=lower_bound, high=upper_bound, size=size))
            
        y = function(self.xs)
        
        poly_xs= create_polynomials(self.xs, N= N)
        
        model = LinearRegression()
        model.fit(poly_xs, y)
        self.coefs = model.coef_
        predictions = model.predict(poly_xs)
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.xs, y, label='Function', marker='o')
        plt.plot(self.xs, predictions, label='Taylor Approximation', marker='o', color='red')
        plt.title("Plot of the Function and its Taylor Series Approximation")
        plt.xlim(lower_bound *1.1, upper_bound *  1.1)
        plt.ylim(np.min(y) - abs(0.1 *np.max(y)), np.max(y) * 1.1)
        plt.text(x = lower_bound, y = np.min(y)+0.2*(np.max(y)-np.min(y)), s = f"Order = {N}")
        if func is str:
            plt.text(x = lower_bound, y = np.min(y)+0.4*(np.max(y)-np.min(y)), s = f"Function = Custom Function")
        else:
            plt.text(x = lower_bound, y = np.min(y)+0.4*(np.max(y)-np.min(y)), s = f"Function = {func}")
        plt.legend()
        plt.show()
        
        self.ys = predictions
        
    def visualize_plots(self, N=10, A = 0, func = "exponential"):
        for i in range(N+1):
            self.plot_function(func = func, N = i)
            display(plt.gcf())
            plt.close()
            time.sleep(0.5)
            clear_output(wait=True) 
