import matplotlib.pyplot as plt

x_train = [1.0, 1.7, 2.0, 2.5, 3.0, 3.2]
y_train = [230, 330, 490, 500, 650, 700]

def gradient_descent(x, y, w, b):
    alpha = 0.01
    while True:
        dj_db = dj_dw = 0
        for i in range(len(x)):
            sum = w * x[i] + b - y[i]
            dj_db += sum
            dj_dw += x[i] * sum
        new_w = w - alpha * dj_dw / len(x)
        new_b = b - alpha * dj_db / len(x)
        if w == new_w and b == new_b: break
        w = new_w
        b = new_b
    return w, b

def compute_model_output(x, w, b):
    f_wb = []
    for i in range(len(x)):
        f_wb.append(w * x[i] + b)
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
