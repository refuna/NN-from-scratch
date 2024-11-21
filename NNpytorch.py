import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network class
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        # Define the network architecture with three fully connected layers
        self.fc1 = nn.Linear(input_size, 64)  # Input layer to hidden layer 1 with 64 neurons
        self.fc2 = nn.Linear(64, 128)        # Hidden layer 1 to hidden layer 2 with 128 neurons
        self.fc3 = nn.Linear(128, output_size)  # Hidden layer 2 to output layer
        self.relu = nn.ReLU()  # Activation function: ReLU

    def forward(self, x):
        # Forward pass through the network
        out = self.fc1(x)  # Pass through the first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Pass through the second layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc3(out)  # Pass through the third (output) layer
        return out

# Define the input and output dimensions
input_size = 1
output_size = 1

# Define the number of training epochs and the learning rate
num_epochs = 1000
learning_rate = 0.01

# create training data
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 100)  # 100 points evenly spaced between -2π and 2π
y_train = np.sin(x_train)  # Compute the sine of the training points
x_train = torch.Tensor(x_train).view(-1, 1)  # Convert training data to PyTorch tensors and reshape to column vectors
y_train = torch.Tensor(y_train).view(-1, 1)

# Initialize the neural network, loss function, and optimizer
net = Net(input_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error as the loss function
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
for epoch in range(num_epochs):
    # Forward pass: Compute the network's predictions
    outputs = net(x_train)
    loss = criterion(outputs, y_train)  # Compute the loss (difference between prediction and target)

    # Backward pass and optimization
    optimizer.zero_grad()  # Reset the gradients
    loss.backward()  # Compute gradients using backpropagation
    optimizer.step()  # Update network weights

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate test data
x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)  # 1000 test points between -2π and 2π
x_test = torch.Tensor(x_test).view(-1, 1)  # Convert to PyTorch tensors and reshape to column vectors

# Prediction on the test data
net.eval()  # Set the network to evaluation mode (no gradient computation)
with torch.no_grad():  # Disable gradient tracking
    y_pred = net(x_test)  # Predict the output for test data

# Visualization of the results
plt.figure(figsize=(10, 6))
x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 1000)  # Points for plotting the sine function
y_plot = np.sin(x_plot)  # Compute the sine values
plt.plot(x_plot, y_plot, color='g', label='sin(x)')  # Plot the sine function
plt.scatter(x_train.numpy(), y_train.numpy(), color='b', label='Training data')  # Plot the training data points
plt.scatter(x_test.numpy(), y_pred.numpy(), color='r', label='Predicted points')  # Plot the predicted points
plt.xlabel('x')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis
plt.title('Fitting sin(x) with Neural Network')  # Title of the plot
plt.legend()  # Add legend to the plot
plt.show()  # Display the plot
