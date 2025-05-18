import torch
from torch import nn
from cnn import*
import random

# Load datasets into the script
train_X, train_Y = read_npz("data/train.npz")
test_X, test_Y = read_npz("data/test.npz")

# Initialise CNN network and optimize weights by gradient descent
model = CNN(8, 8, 10, 10, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(model)

# Hyperparameter settings
learning_rate = 0.01
epochs = 10
batch_size = 30

number_of_training_samples = train_X.shape[0]
number_of_test_samples = test_X.shape[0]

# Training loop
print("Training")
for epoch in range(epochs):
    train_loss = 0.
    train_accuracy = 0.

    for i in range(number_of_training_samples // batch_size):
        X_i = train_X[None, i : i + batch_size, :]
        y_i = train_Y[i : i + batch_size]

        # Forward pass through network
        y_hat, _, _ = model(X_i)

        # Compute loss
        loss = loss_fn(y_hat, y_i)

        # Backprop through network
        optimizer.zero_grad()
        loss.backward()

        # Weight update / weight optimization
        optimizer.step()
        
        with torch.no_grad():
            train_loss += loss
            train_accuracy += hits(y_hat,y_i)
                      
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for i in range(number_of_test_samples//100):
            X_i = test_X[None, i:i + 100, :]
            y_i = test_Y[i:i + 100]

            # Forward pass through network
            y_hat, _, _ = model(X_i)

            # Compute loss
            loss = loss_fn(y_hat, y_i)

            optimizer.zero_grad()

            test_loss += loss
            test_accuracy += hits(y_hat,y_i)

    train_loss /= number_of_training_samples
    train_accuracy /= number_of_training_samples / 100
    test_loss /= number_of_test_samples
    test_accuracy /= number_of_test_samples / 100
    
    print(f"Epoch {epoch + 1} Loss: {train_loss} Accuracy: {train_accuracy} " +
          f"Test Loss: {test_loss} Test Accuracy: {test_accuracy}")
    
# Plots the CNN network's filters using a random sample from the test set
random_sample = random.randint(0, test_X.shape[0])
X = test_X[random_sample, :]
y_hat, z_1, v_1 = model(X)

plot_filters(X=X, H=model.conv_1.weight, z_1=z_1, v_1=v_1)
