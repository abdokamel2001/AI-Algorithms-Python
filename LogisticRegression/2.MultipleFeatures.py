import copy
import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
initial_w = np.zeros(2)
initial_b = 0.0
alpha = 0.1
iterations = 10000

def sigmoid(z): return 1/(1+np.exp(-z))

def compute_cost(x, y, w, b):
    m,n = x.shape
    cost = 0.0
    for i in range(m):
        f_wb_i = b
        for j in range(n):
            f_wb_i += x[i][j] * w[j]
        f_wb_i = sigmoid(f_wb_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost /= m
    return cost

def compute_gradient(x, y, w, b): 
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        f_wb = b
        for j in range(n):
            f_wb += x[i][j] * w[j]
        f_wb = sigmoid(f_wb)
        err  = f_wb  - y[i]
        for j in range(n):
            dj_dw[j] += x[i][j] * err
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(X, y, w, b)   
        w -= alpha * dj_dw               
        b -= alpha * dj_db               
        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))
        if i% math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history

w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = x_train.shape
for i in range(m):
    print(
f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()
