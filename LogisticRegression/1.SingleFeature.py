import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0, 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0, 0, 0, 1, 1, 1],dtype=np.longdouble)
w_in = np.zeros((1))
b_in = 0

def sigmoid(z): return 1/(1+np.exp(-z))

def gradient_descent(x, y, w, b):
    alpha = 0.01
    Precision = 10**-5
    while True:
        dj_db = dj_dw = 0
        for i in range(len(x)):
            sum = sigmoid(w * x[i] + b) - y[i]
            dj_db += sum
            dj_dw += x[i] * sum
        new_w = w - alpha * dj_dw / len(x)
        new_b = b - alpha * dj_db / len(x)
        if (abs(new_w - w) <= Precision) and (
            abs(new_b - b) <= Precision): break
        w = new_w
        b = new_b
    return w, b

def compute_model_output(x, w, b):
    f_wb = []
    for i in range(len(x)):
        f_wb.append(sigmoid(w * x[i] + b))
    return f_wb

w, b = gradient_descent(x_train, y_train, 0, 0)
tmp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title(f"y = {w:.2f}x + {b:.2f}")
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.show()
