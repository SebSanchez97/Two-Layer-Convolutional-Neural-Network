# Project Description: Custom Convolutional Neural Network (CNN) for MNIST Digit Classification

## Data Source:
This project uses the MNIST dataset, a well-known benchmark dataset for image classification. The dataset is stored in two NumPy .npz files:

Training Set (train.npz):

Features (X): 60,000 samples, each with 784 features (28x28 flattened grayscale images).

Labels (Y): One-hot encoded class labels for 10 classes (digits 0-9).

Test Set (test.npz):

Features (X): 10,000 samples, each with 784 features (28x28 flattened grayscale images).

Labels (Y): One-hot encoded class labels for 10 classes (digits 0-9).

# Data Processing:
The dataset is loaded using the read_npz function in cnn.py, which reads the .npz files and extracts the feature vectors (X) and one-hot encoded labels (Y).

The feature vectors (28x28 images) are flattened into 784-dimensional vectors for input into the model.

The labels are one-hot encoded to match the 10-digit classification problem (0-9).

# Model Architecture:

This project implements a custom Convolutional Neural Network (CNN) for image classification:

The model is defined in cnn.py and uses PyTorch's nn.Module for model creation.

Input Layer: Accepts 784-dimensional flattened image vectors.

Convolutional Layers: Two convolutional layers with:

First convolution: 8 filters (kernels).

Second convolution: 8 filters (kernels).

Fully Connected Layers: Two fully connected (dense) layers:

First dense layer: 10 neurons.

Second dense layer: 10 neurons (output layer).

Activation Function: ReLU for hidden layers and Softmax for the output layer.

Output Layer: 10 neurons, each representing the probability of one of the 10 digit classes (0-9).

# Training Process:
The model is trained using the training set with the following settings:

Loss Function: Cross-Entropy Loss (suitable for multi-class classification).

Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.

Batch Size: 30 samples per batch.

Number of Epochs: 10.

The training loop in main.py performs the following:

For each epoch:

The model performs forward propagation, calculates the loss, and backpropagates the error.

The model weights are updated using gradient descent.

# Evaluation:
After training, the model is evaluated on the test set.

The model's accuracy is measured, and the decision boundaries can be visualized using matplotlib.

# Visualization:
The training and test samples can be visualized as 28x28 grayscale images.

The model's predictions can be visualized using confusion matrices and accuracy plots.