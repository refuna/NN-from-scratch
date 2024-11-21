import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        # Initialize input size, hidden layer size, and output size
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        # Initialize weights and biases with random values
        np.random.seed(0)
        self.w1 = np.random.randn(hidden_layer_size, input_size)  # Weights for input to hidden layer
        self.b1 = np.random.randn(hidden_layer_size)  # Bias for hidden layer
        self.w2 = np.random.randn(output_size, hidden_layer_size)  # Weights for hidden to output layer
        self.b2 = np.random.randn(output_size)  # Bias for output layer

    # Forward propagation
    def forward(self, x):
        # Compute the hidden layer activations using the input and weights
        hidden_layer = np.dot(x.reshape(-1, 1), self.w1.T) + self.b1.reshape(1, -1)
        hidden_layer = np.maximum(0, hidden_layer)  # Apply ReLU activation
        # Compute the predicted output using the hidden layer and weights
        predicted_output = np.dot(self.w2, hidden_layer.T) + self.b2.reshape(-1, 1)
        return hidden_layer, predicted_output

    # Backward propagation to compute gradients
    def backward(self, x, y, hidden_layer, predicted_output):
        # Compute the gradient of the loss with respect to the predicted output
        grad_predicted_output = 2 * (predicted_output - y) / len(x)
        # Compute gradients for weights and biases in the second layer (hidden to output)
        grad_w2 = np.dot(grad_predicted_output, hidden_layer)
        grad_b2 = np.sum(grad_predicted_output, axis=1)
        # Compute ReLU derivative for the hidden layer
        relu_derivative = np.where(np.dot(x.reshape(-1, 1), self.w1.T) + self.b1.reshape(1, -1) > 0, 1, 0)
        # Compute gradients for the hidden layer
        grad_hidden_layer = np.dot(self.w2.T, grad_predicted_output) * relu_derivative.T
        # Compute gradients for weights and biases in the first layer (input to hidden)
        grad_w1 = np.dot(grad_hidden_layer, x.reshape(-1, 1))
        grad_b1 = np.sum(grad_hidden_layer, axis=1)

        return grad_w1, grad_b1, grad_w2, grad_b2

    # Update parameters using gradients and learning rate
    def update(self, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate):
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2

    # Train the neural network
    def train(self, x, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward propagation
            hidden_layer, predicted_output = self.forward(x)
            # Compute the loss
            loss = np.square(predicted_output - y).mean()
            # Print the loss at every 100th epoch
            if epoch % 100 == 0:
                print(f"epoch {epoch}: loss = {loss}")
            # Backward propagation to compute gradients
            grad_w1, grad_b1, grad_w2, grad_b2 = self.backward(x, y, hidden_layer, predicted_output)
            # Update weights and biases
            self.update(grad_w1, grad_b1, grad_w2, grad_b2, learning_rate)

    # Predict using the trained model
    def predict(self, x):
        _, predicted_output = self.forward(x)
        return predicted_output

if __name__ == '__main__':
    # Generate training data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)

    # Create a neural network object
    net = NeuralNetwork(input_size=1, hidden_layer_size=64, output_size=1)

    # Train the network
    learning_rate = 0.001
    num_epochs = 10000
    net.train(x=x, y=y, learning_rate=learning_rate, num_epochs=num_epochs)

    # Test the model
    test_x = np.linspace(0, 2 * np.pi, 100)
    test_y = np.cos(test_x)
    predicted_output = net.predict(test_x)
    test_loss = np.square(predicted_output - test_y).mean()
    print(f"Test Loss: {test_loss}")

    # Plot the original cosine function and the model's predictions
    plt.scatter(test_x, test_y, label='cos(x)')
    plt.scatter(test_x, predicted_output.T, label='Fitted', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting cos(x)')
    plt.legend()
    plt.show()
