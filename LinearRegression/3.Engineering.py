import math
import numpy as np
import matplotlib.pyplot as plt

num_iters = 1000000
alpha = 1e-1

def zscore_normalize_features(X):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      
    return X_norm

def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i],w) + b       
        cost += (f_wb_i - y[i])**2              
    cost /= (2*m)                                 
    return(np.squeeze(cost)) 

def compute_gradient(X, y, w, b): 
    m = X.shape[0]
    f_wb = np.dot(X,w) + b              
    e   = f_wb - y                
    dw  = (1/m) * np.dot(X.T, e)
    db  = (1/m) * np.sum(e) 
    return db,dw

def gradient_descent(X, y, w, b): 
    for i in range(num_iters):
        dj_db,dj_dw = compute_gradient(X, y, w, b)   
        w -= alpha * dj_dw               
        b -= alpha * dj_db
        if i % math.ceil(num_iters/10) == 0:
            cst = compute_cost(X, y, w, b)
            print(f"Iteration {i:9d}, Cost: {cst:0.5e}")
    return w, b

def run_gradient_descent_feng(X,y):
    n = X.shape[1]
    w = np.zeros(n)
    b = 0
    w_out, b_out = gradient_descent(X ,y, w, b)  
    return(w_out, b_out)


x = np.arange(0,20,1)
y = np.cos(x/3)
X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 
model_w,model_b = run_gradient_descent_feng(X, y)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,np.dot(X, model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
