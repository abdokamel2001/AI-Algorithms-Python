import numpy as np

iterations = 10000
alpha = 5.0e-7
lambda_ = 0.1

def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])
    b = 0.0
    return w, b

def compute(w, b, X, Y):
    m = X.shape[1]
    A = np.dot(w.T, X) + b - Y
    cost = 0.5 / m * np.dot(A, A)
    dw = 1 / m * np.dot(X, A.T)
    db = 1 / m * np.sum(A)
    cost = np.squeeze(np.array(cost))
    
    #Regulization
    reg_cost = np.dot(w, w)
    reg_cost *= 0.5 * lambda_ / m
    total_cost = cost + reg_cost
    dw += w * lambda_ / m
    
    grads = {"dw": dw, "db": db}
    return grads, total_cost

def gradient_descent(w, b, X, Y):
    costs = []
    for i in range(iterations):
        grads, cost = compute(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w -= dw * alpha
        b -= db * alpha
        if i % 100 == 0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,"b": b}
    grads = {"dw": dw,"db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    Y = np.dot(w.T, X) + b 
    return Y
