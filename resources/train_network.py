import numpy as np
import pandas as pd

data = pd.read_csv("mnist_train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train=data.T
Ytrain = data_train[0]
Xtrain = data_train[1:n]
Xtrain = Xtrain / 255.0

data_test = pd.read_csv("mnist_test.csv")
data_test = np.array(data_test)
m_test, n_test = data_test.shape
np.random.shuffle(data_test)  

data_dev = data_test.T
Ydev = data_dev[0]
Xdev = data_dev[1:n_test]  
Xdev = Xdev / 255.0

def init_params(n, input, output):
    params = {}
    layers = [input] + n + [output]
    
    for i in range(1, len(layers)):
        params[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2. / layers[i-1])
        params[f'b{i}'] = np.zeros((layers[i], 1))
    
    return params


def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(params, X):
    cache = {}
    L = len(params) // 2
    
    cache['Z1']=params['W1'].dot(X)+params['b1']
    cache['A1']=ReLU(cache['Z1'])
    
    for i in range(2, L):
        cache[f'Z{i}'] = params[f'W{i}'].dot(cache[f'A{i-1}']) + params[f'b{i}']
        cache[f'A{i}'] = ReLU(cache[f'Z{i}'])
        
    cache[f'Z{L}'] = params[f'W{L}'].dot(cache[f'A{L-1}']) + params[f'b{L}']
    cache[f'A{L}'] = softmax(cache[f'Z{L}'])
    
    return cache 

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(params, cache, X, Y):
    grads = {}
    m=Y.size
    L=len(params)//2
    onehotY = one_hot(Y)
        
    dZ = cache[f'A{L}'] - onehotY
    grads[f'dW{L}'] = 1/m * dZ.dot(cache[f'A{L-1}'].T)
    grads[f'db{L}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    
    for l in range(L-1, 0, -1):
        dA = params[f'W{l+1}'].T.dot(dZ)
        dZ = dA * deriv_ReLU(cache[f'Z{l}'])
        A_prev = cache[f'A{l-1}'] if l > 1 else X
        grads[f'dW{l}'] = 1 / m * dZ.dot(A_prev.T)
        grads[f'db{l}'] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    return grads

def update_params(params, grads, alpha):
    L = len(params)//2
    
    for l in range(1, L + 1):
        params[f'W{l}'] = params[f'W{l}'] - alpha * grads[f'dW{l}']
        params[f'b{l}'] = params[f'b{l}'] - alpha * grads[f'db{l}']
        
    return params

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha, n, input, output): 
    
    params = init_params(n, input, output)
    for i in range(iterations):
        cache = forward_prop(params, X)
        grads = back_prop(params, cache, X, Y)
        params = update_params(params, grads, alpha)
        
        if i % 10 == 0:
            L = len(params)//2
            predictions = get_predictions(cache[f'A{L}'])
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}")
            print(f"Accuracy: {accuracy}")
            
    return params

params = gradient_descent(Xtrain, Ytrain, 5000, 0.1, [64, 32], 784, 10)

L = len(params)//2
cache = forward_prop(params, Xdev)
predictions = get_predictions(cache[f'A{L}'])
print(get_accuracy(predictions, Ydev))
np.save('mnist_model_params.npy', params)