# MNIST-neural-network
This collection of scripts supports the development and deployment of a neural network model, optimised for recognizing handwritten digits from the MNIST dataset. It includes training, parameter conversion for frontend integration, realtime prediction, and visualization tools.

## Contents

- `1_layer.py`: Implementation of a single-layer neural network for MNIST digit classification.
- `train_network.py`: Script to train the neural network model using the MNIST dataset.
- `train_network(with_aug).py`: Training script with data augmentation for improved model performance.
- `npy_to_json.py`: Utility to convert NumPy model parameters to JSON format for frontend or other uses.
- `realtime.py`: Script for realtime digit prediction using the trained model.
- `viewer2.py`: Visualization of the MNIST augmentation output from `train_network(with_aug).py`.
- `index.html`: Frontend web interface for the real-time MNIST digit predictor.

## Usage

1. Train the model using `train_network.py` or `train_network(with_aug).py` for augmented training.
2. Convert the trained model parameters to JSON using `npy_to_json.py` for frontend integration.
3. Use `realtime.py` for realtime digit prediction.
4. Visualize or analyze results with `viewer2.py`.
5. Open `index.html` in a browser to use the web-based real-time MNIST digit predictor interface.
