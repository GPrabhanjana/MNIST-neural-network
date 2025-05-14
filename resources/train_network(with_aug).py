import numpy as np
import pandas as pd
from scipy import ndimage

def augment_image(X, img_size=28, 
                  max_rotation=15,  
                  max_scale_pct=7,  
                  max_offset_pct=10,  
                  max_noise_pct=3):  
    
    is_normalized = X.max() <= 1.0
    scale_factor = 1.0 if is_normalized else 255.0
    
    augmented_X = np.zeros_like(X)
    max_scale = max_scale_pct / 100.0
    max_offset = int(img_size * (max_offset_pct / 100.0))
    noise_amp = max_noise_pct / 100.0 * scale_factor

    for i in range(X.shape[1]):
        img = X[:, i].reshape(img_size, img_size)
        augmented = img.copy().astype(float)

        # Random rotation
        angle = np.random.uniform(-max_rotation, max_rotation)
        augmented = ndimage.rotate(augmented, angle, reshape=False, mode='constant', cval=0)

        # Random scaling
        scale = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)
        if scale != 1.0:
            scaled = ndimage.zoom(augmented, scale, mode='constant', cval=0)
            if scale > 1.0:
                start = (scaled.shape[0] - img_size) // 2
                augmented = scaled[start:start+img_size, start:start+img_size]
            else:
                pad = (img_size - scaled.shape[0]) // 2
                padded = np.zeros((img_size, img_size))
                padded[pad:pad+scaled.shape[0], pad:pad+scaled.shape[1]] = scaled
                augmented = padded

        # Random shift
        offset_x = np.random.randint(-max_offset, max_offset + 1)
        offset_y = np.random.randint(-max_offset, max_offset + 1)
        shifted = np.zeros_like(augmented)

        src_y_start = max(0, -offset_y)
        src_y_end = min(img_size, img_size - offset_y)
        src_x_start = max(0, -offset_x)
        src_x_end = min(img_size, img_size - offset_x)

        dst_y_start = max(0, offset_y)
        dst_y_end = min(img_size, img_size + offset_y)
        dst_x_start = max(0, offset_x)
        dst_x_end = min(img_size, img_size + offset_x)

        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = augmented[src_y_start:src_y_end, src_x_start:src_x_end]
        augmented = shifted

        # Add noise (adjusted based on normalization)
        noise = np.random.normal(0, noise_amp / 3, size=augmented.shape)
        salt_mask = np.random.random(augmented.shape) < (max_noise_pct / 500.0)
        pepper_mask = np.random.random(augmented.shape) < (max_noise_pct / 500.0)

        max_val = 1.0 if is_normalized else 255.0
        augmented[salt_mask] = np.minimum(augmented[salt_mask] + noise_amp, max_val)
        augmented[pepper_mask] = np.maximum(augmented[pepper_mask] - noise_amp, 0)
        augmented = np.clip(augmented + noise, 0, max_val)

        # Store the result
        augmented_X[:, i] = augmented.reshape(-1)

    return augmented_X

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

# Load and prepare data
data = pd.read_csv("mnist_train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  

data_train = data.T
Ytrain = data_train[0]
Xtrain = data_train[1:n]

# Normalization
Xtrain = Xtrain / 255.0

augmentation_ratio = 0.9  
split_idx = int(Xtrain.shape[1] * (1 - augmentation_ratio))

# Split data into portions
Xtrain_orig = Xtrain[:, :split_idx]
Xtrain_to_augment = Xtrain[:, split_idx:]
Ytrain_orig = Ytrain[:split_idx]
Ytrain_to_augment = Ytrain[split_idx:]

Xtrain_augmented = augment_image(Xtrain_to_augment)

# Combine original and augmented data
Xtrain = np.concatenate((Xtrain_orig, Xtrain_augmented), axis=1)
Ytrain = np.concatenate((Ytrain_orig, Ytrain_to_augment))

# Shuffle the combined data
shuffle_indices = np.random.permutation(Xtrain.shape[1])
Xtrain = Xtrain[:, shuffle_indices]
Ytrain = Ytrain[shuffle_indices]

# Load and prepare test data
data_test = pd.read_csv("mnist_test.csv")
data_test = np.array(data_test)
m_test, n_test = data_test.shape

data_dev = data_test.T
Ydev = data_dev[0]
Xdev = data_dev[1:n_test]  
Xdev = Xdev / 255.0  # Normalize test data

indices_0 = np.where(Ytrain == 0)[0]
indices_6 = np.where(Ytrain == 6)[0]

sample_0 = np.random.choice(indices_0, size=int(0.2 * len(indices_0)), replace=False)
sample_6 = np.random.choice(indices_6, size=int(0.2 * len(indices_6)), replace=False)

X_0 = Xtrain[:, sample_0]
X_6 = Xtrain[:, sample_6]

# Augment 0: scale down 5–14%, then apply full augmentation
def scale_down_images(X, img_size=28):
    augmented_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        img = X[:, i].reshape(img_size, img_size)
        scale = 1.0 - np.random.uniform(0.05, 0.14)
        scaled = ndimage.zoom(img, scale, mode='constant', cval=0)
        pad = (img_size - scaled.shape[0]) // 2
        padded = np.zeros((img_size, img_size))
        padded[pad:pad+scaled.shape[0], pad:pad+scaled.shape[1]] = scaled
        # Apply rotation, shift, and noise using augment_image logic (simplified)
        augmented = padded
        augmented = ndimage.rotate(augmented, np.random.uniform(-15, 15), reshape=False, mode='constant', cval=0)
        offset_x = np.random.randint(-2, 3)
        offset_y = np.random.randint(-2, 3)
        shifted = np.zeros_like(augmented)
        y1 = max(0, offset_y); y2 = min(img_size, img_size + offset_y)
        x1 = max(0, offset_x); x2 = min(img_size, img_size + offset_x)
        sy1 = max(0, -offset_y); sy2 = min(img_size, img_size - offset_y)
        sx1 = max(0, -offset_x); sx2 = min(img_size, img_size - offset_x)
        shifted[y1:y2, x1:x2] = augmented[sy1:sy2, sx1:sx2]
        noise = np.random.normal(0, 0.01, size=shifted.shape)
        salt = np.random.rand(*shifted.shape) < 0.01
        pepper = np.random.rand(*shifted.shape) < 0.01
        shifted[salt] = 1.0
        shifted[pepper] = 0.0
        shifted = np.clip(shifted + noise, 0, 1.0)
        augmented_X[:, i] = shifted.reshape(-1)
    return augmented_X

# Augment 6: scale up 5–12%, apply only noise
def scale_up_and_noise(X, img_size=28):
    augmented_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        img = X[:, i].reshape(img_size, img_size)
        scale = 1.0 + np.random.uniform(0.05, 0.12)
        scaled = ndimage.zoom(img, scale, mode='constant', cval=0)
        start = (scaled.shape[0] - img_size) // 2
        cropped = scaled[start:start+img_size, start:start+img_size]
        # Add only noise
        noise = np.random.normal(0, 0.01, size=cropped.shape)
        salt = np.random.rand(*cropped.shape) < 0.01
        pepper = np.random.rand(*cropped.shape) < 0.01
        cropped[salt] = 1.0
        cropped[pepper] = 0.0
        cropped = np.clip(cropped + noise, 0, 1.0)
        augmented_X[:, i] = cropped.reshape(-1)
    return augmented_X

X_0_aug = scale_down_images(X_0)
X_6_aug = scale_up_and_noise(X_6)

Xtrain = np.concatenate((Xtrain, X_0_aug, X_6_aug), axis=1)
Ytrain = np.concatenate((Ytrain, np.full(X_0_aug.shape[1], 0), np.full(X_6_aug.shape[1], 6)))

# Final shuffle before proceeding to training
shuffle_indices = np.random.permutation(Xtrain.shape[1])
Xtrain = Xtrain[:, shuffle_indices]
Ytrain = Ytrain[shuffle_indices]

# Train the model
print("Training model with augmented data...")
params = gradient_descent(Xtrain, Ytrain, 3000, 0.1, [100, 100], 784, 10)

# Evaluate on test data
L = len(params)//2
cache = forward_prop(params, Xdev)
predictions = get_predictions(cache[f'A{L}'])
test_accuracy = get_accuracy(predictions, Ydev)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
np.save('mnist_model_params.npy', params)