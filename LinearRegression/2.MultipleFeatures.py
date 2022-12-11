import copy
import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

initial_w = np.zeros(4)
initial_b = 0.0
iterations = 10000
alpha = 5.0e-7

def compute_cost(x, y, w, b):
    m, n = x.shape
    cost = 0.0
    for i in range(m):
        f_wb_i = b
        for j in range(n):
            f_wb_i += x[i][j] * w[j]
        cost += (f_wb_i - y[i])**2
    cost /= 2*m
    return cost

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        err = b - y[i]
        for j in range(n):
            err += x[i][j] * w[j]
        for j in range(n):
            dw[j] += x[i][j] * err
        db += err
    dw /= m
    db /= m
    return db, dw

def gradient_descent(x, y, w_in, b_in):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(iterations):
        db, dw = compute_gradient(x, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
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
