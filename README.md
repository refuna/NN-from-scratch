# Project Description: Neural Network for Cosine Function Approximation  

This project demonstrates the implementation of a simple **fully connected neural network** to approximate the cosine function \( \cos(x) \). It is designed as a hands-on example of creating, training, and testing a neural network from scratch using Python and libraries like **NumPy** and **Matplotlib**.  

## Objectives  
The primary objectives of this project include:  
1. **Implementing a neural network architecture** with one hidden layer and using basic mathematical operations for forward and backward propagation.  
2. **Approximating a non-linear function** (\( \cos(x) \)) by training the network on a generated dataset.  
3. Applying techniques like **ReLU activation** and gradient-based optimization to minimize the mean squared error (MSE) between the predicted output and the target values.  
4. **Visualizing the results** to compare the original cosine function and the fitted predictions produced by the trained model.  

## Key Features  
### Neural Network Architecture  
- **Input Layer**: Accepts a single input feature.  
- **Hidden Layer**: Includes 64 neurons with ReLU activation.  
- **Output Layer**: Produces a single output to approximate the cosine function.  

### Training Process  
- **Forward Propagation**: Computes predictions based on the input data.  
- **Loss Function**: Uses Mean Squared Error (MSE) to quantify the difference between predictions and the true values.  
- **Backpropagation**: Computes gradients for weights and biases based on the loss.  
- **Gradient Descent**: Updates model parameters using a configurable learning rate.  

### Data  
- **Input Data (\(x\))**: A set of 100 evenly spaced points between 0 and \(2\pi\).  
- **Target Output (\(y\))**: The cosine values of the input points (\( \cos(x) \)).  

### Visualization  
- Plots the original cosine function alongside the neural network's predictions to evaluate how well the model approximates the function.  

## Tools and Libraries Used  
- **NumPy**: For numerical operations such as matrix multiplication and random initialization of weights and biases.  
- **Matplotlib**: For visualizing the original cosine function and the neural network's predictions.  


