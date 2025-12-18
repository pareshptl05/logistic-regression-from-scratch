import numpy as np
from .activations import sigmoid
from .losses import binary_cross_entropy

class LogisticRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        
        for i in range(self.n_iters):
            z = X @ self.w + self.b
            y_hat = sigmoid(z)

            dw = (X.T @ (y_hat-y))/m
            db = (np.sum(y_hat-y))/m

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            if i%100 == 0:
                print(f"loss: {binary_cross_entropy(y,y_hat)}")
            
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)
    
    def predict(self, X, threshold = 0.5):
        return (self.predict_proba(X)>=threshold).astype(int)