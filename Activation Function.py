import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Determine the number of classes in your dataset
num_classes = len(np.unique(y))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Define the Ada-Act activation function
def ada_act_activation(x, k0, k1):
    return k0 + k1 * x

# Initialize model parameters
k0 = np.random.randn()
k1 = np.random.randn()

# Define neural network architecture
input_size = X.shape[1]
hidden_size = 10
output_size = len(np.unique(y))

# Initialize weights and biases
weights_1 = np.random.randn(input_size, hidden_size)
biases_1 = np.zeros(hidden_size)
weights_2 = np.random.randn(hidden_size, output_size)
biases_2 = np.zeros(output_size)

# Training loop
num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_train, weights_1) + biases_1
    hidden_layer_output = ada_act_activation(hidden_layer_input, k0, k1)
    output_layer_input = np.dot(hidden_layer_output, weights_2) + biases_2
    output_probs = softmax(output_layer_input)

    # Convert y_train to one-hot encoded format
    y_train_onehot = to_categorical(y_train, num_classes=output_size)
    
    # Calculate loss
    loss = categorical_cross_entropy(y_train_onehot, output_probs)
    
    # Backpropagation
    # Calculate gradients
    
    # Update parameters
    # Update weights and biases
    
    # Print loss for tracking progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


#result - loss function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the number of classes in your dataset
num_classes = len(np.unique(y))

# Convert y_train to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)

# Define the Ada-Act activation function
def ada_act_activation(x, k0, k1):
    return k0 + k1 * x

# Initialize model parameters
k0 = np.random.randn()
k1 = np.random.randn()

# Define neural network architecture
input_size = X.shape[1]
hidden_size = 10
output_size = num_classes

# Initialize weights and biases
weights_1 = np.random.randn(input_size, hidden_size)
biases_1 = np.zeros(hidden_size)
weights_2 = np.random.randn(hidden_size, output_size)
biases_2 = np.zeros(output_size)

# Initialize arrays to store losses and epochs
train_losses = []
epochs = []

# Training loop
num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_train, weights_1) + biases_1
    hidden_layer_output = ada_act_activation(hidden_layer_input, k0, k1)
    output_layer_input = np.dot(hidden_layer_output, weights_2) + biases_2
    output_probs = softmax(output_layer_input)

    # Convert y_train to one-hot encoded format
    y_train_onehot = to_categorical(y_train, num_classes=output_size)
    
    # Calculate loss
    loss = categorical_cross_entropy(y_train_onehot, output_probs)
    train_losses.append(loss)
    epochs.append(epoch)
    
    # Backpropagation
    # Calculate gradients
    
    # Update parameters
    # Update weights and biases
    
    # Print loss for tracking progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Plot loss vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Progression')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the number of classes in your dataset
num_classes = len(np.unique(y))

# Convert y_train and y_test to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Define the Ada-Act activation function
def ada_act_activation(x, k0, k1):
    return k0 + k1 * x

# Initialize model parameters
k0 = np.random.randn()
k1 = np.random.randn()

# Define neural network architecture
input_size = X.shape[1]
hidden_size = 10
output_size = num_classes

# Initialize weights and biases
weights_1 = np.random.randn(input_size, hidden_size)
biases_1 = np.zeros(hidden_size)
weights_2 = np.random.randn(hidden_size, output_size)
biases_2 = np.zeros(output_size)

# Initialize arrays to store losses and epochs
train_losses = []
test_losses = []
epochs = []

# Training loop
num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward propagation for training
    hidden_layer_input_train = np.dot(X_train, weights_1) + biases_1
    hidden_layer_output_train = ada_act_activation(hidden_layer_input_train, k0, k1)
    output_layer_input_train = np.dot(hidden_layer_output_train, weights_2) + biases_2
    output_probs_train = softmax(output_layer_input_train)
    train_loss = categorical_cross_entropy(y_train_onehot, output_probs_train)
    train_losses.append(train_loss)
    
    # Forward propagation for testing
    hidden_layer_input_test = np.dot(X_test, weights_1) + biases_1
    hidden_layer_output_test = ada_act_activation(hidden_layer_input_test, k0, k1)
    output_layer_input_test = np.dot(hidden_layer_output_test, weights_2) + biases_2
    output_probs_test = softmax(output_layer_input_test)
    test_loss = categorical_cross_entropy(y_test_onehot, output_probs_test)
    test_losses.append(test_loss)
    
    epochs.append(epoch)
    
    # Backpropagation
    # Calculate gradients
    
    # Update parameters
    # Update weights and biases
    
    # Print loss for tracking progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plot train and test losses vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs. Test Loss')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the number of classes in your dataset
num_classes = len(np.unique(y))

# Convert y_train and y_test to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# ... Define other functions and parameters ...

# Initialize arrays to store losses and epochs
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
epochs = []

# Training loop
num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward propagation for training
    # ... Calculate train loss ...

    # Forward propagation for testing
    # ... Calculate test loss ...

    # Calculate train accuracy
    train_predictions = np.argmax(output_probs_train, axis=1)
    train_accuracy = np.mean(train_predictions == y_train)
    train_accuracies.append(train_accuracy)
    
    # Calculate test accuracy
    test_predictions = np.argmax(output_probs_test, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)
    test_accuracies.append(test_accuracy)

    epochs.append(epoch)

    # ... Backpropagation and parameter updates ...

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# ... Plot loss and accuracy graphs ...

# Final accuracy values
final_train_accuracy = train_accuracies[-1]
final_test_accuracy = test_accuracies[-1]

print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the number of classes in your dataset
num_classes = len(np.unique(y))

# Convert y_train and y_test to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# ... Define other functions and parameters ...

# Initialize arrays to store losses, accuracies, and epochs
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_f1_scores = []
test_f1_scores = []
epochs = []

# Training loop
num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward propagation for training
    # ... Calculate train loss ...

    # Forward propagation for testing
    # ... Calculate test loss ...

    # Calculate train accuracy and F1-Score
    train_predictions = np.argmax(output_probs_train, axis=1)
    train_accuracy = np.mean(train_predictions == y_train)
    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    train_accuracies.append(train_accuracy)
    train_f1_scores.append(train_f1)
    
    # Calculate test accuracy and F1-Score
    test_predictions = np.argmax(output_probs_test, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_accuracies.append(test_accuracy)
    test_f1_scores.append(test_f1)

    epochs.append(epoch)

    # ... Backpropagation and parameter updates ...

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Train F1-Score: {train_f1:.4f}, Test F1-Score: {test_f1:.4f}")

# ... Plot loss and accuracy graphs ...

# Final accuracy and F1-Score values
final_train_accuracy = train_accuracies[-1]
final_test_accuracy = test_accuracies[-1]
final_train_f1 = train_f1_scores[-1]
final_test_f1 = test_f1_scores[-1]

print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
print(f"Final Train F1-Score: {final_train_f1:.4f}")
print(f"Final Test F1-Score: {final_test_f1:.4f}")
