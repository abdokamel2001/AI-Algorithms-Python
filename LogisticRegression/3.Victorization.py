import numpy as np

iterations = 10000
alpha = 0.1

def sigmoid(z): return 1/(1+np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])
    b = 0.0
    return w, b

def compute(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * (np.dot(Y, np.log(A).T) +
                     np.dot((1-Y), np.log(1 - A).T))
    dw = 1 / m * (np.dot(X, (A - Y).T))
    db = 1 / m * (np.sum(A - Y))
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

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
    grads = {"dw": dw ,"db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b)    
    for i in range(A.shape[1]):
        if(A[0][i] <= 0.5):
            Y[0][i] = 0
        else:
            Y[0][i] = 1
    return Y
