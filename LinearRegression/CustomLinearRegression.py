import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, lr, l1, epoch, init='random'):
        """
        Gradient Descent Linear Regression with optional L1 Lasso Regularization Techniques

        :param lr: Learning Rate (Recommended: 0.001)
        :param l1: L1 Lasso Regularization Rate (Recommended: 0.001)
        :param epoch: Number of iterations for training (Recommended: 2000)
        :param init: Initialization Method for weights and biases at the beginning (Recommended: 'random')
        """
        self.lr = lr
        self.l1 = l1
        self.epoch = epoch
        self.init = init
        self.weights = None
        self.bias = None

    def loss(self, pred, actual):
        m, _ = actual.shape
        mse = np.mean(np.square(pred - actual)) # Calculate the mean square error loss function
        l1_penalty = self.l1 * np.sum(np.abs(self.weights)) / m # Calculate the lasso penalty value and average it by the row numbers
        # By combining both mse and l1 regularisation we get the final loss, l1 is used to prevent overfitting in continuous algorithms like linear regression
        return mse + l1_penalty

    def fit(self, x_train, y_train, x_test, y_test):
        # Convert x and y from original datatypes to numpy arrays
        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        y_train = np.asarray(y_train).reshape(-1, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)

        # Initiate the shape attributes for x_train (m for rows, n for columns)
        m, n = x_train.shape

        # Initialize weights and biases
        # Method 1: You can make the weight and bias value as random
        if self.init == 'random':
            self.weights = np.random.randn(n, 1) / np.sqrt(n) # You divide by sqrt of the weights to normalize it so the gradient doesn't explode
            self.bias = np.random.randn(1, 1)
        # Method 2: You can start the weight and bias from 0
        elif self.init == 'zeros':
            self.weights = np.zeros((n, 1))
            self.bias = np.zeros((1, 1))
        # Else call error
        else:
            raise ValueError(f'Invalid init method {self.init}, init must be random or zeros!')

        # Declare losses
        train_losses = []
        test_losses = []

        for _ in range(self.epoch):
            train_pred = np.dot(x_train, self.weights) + self.bias # Forward propagation for training
            train_loss = self.loss(train_pred, y_train) # Computing training loss
            train_losses.append(train_loss) # Track training loss record

            test_pred = np.dot(x_test, self.weights) + self.bias # Forward propagation for testing
            test_loss = self.loss(test_pred, y_test) # Computing testing loss
            test_losses.append(test_loss) # Track test loss record

            error = train_pred - y_train # Calculate the difference in predicted and actual value
            dW = 2/m * np.dot(x_train.T, error) + (self.l1/m) * np.sign(self.weights) # Compute the derivative of loss function with respect to weights
            db = 2/m * np.sum(error, axis=0, keepdims=True) # Compute the derivative of loss function with respect to bias

            # Gradient descent formulas
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            # Print training and testing loss per 500 epochs
            if _ % 500 == 0:
                print(f"Epoch: {_} | Train loss: {train_loss} | Test loss: {test_loss}")

        # Plot the train test loss curve
        plt.plot(range(self.epoch), train_losses, color='red', label='Train loss') # Plot for train loss
        plt.plot(range(self.epoch), test_losses, color='blue', linestyle='dashed', label='Test loss') # Plot for test loss
        plt.xlabel('Epoch') # Label x-axis
        plt.ylabel('Loss') # Label y-axis
        plt.title('Loss vs Epoch') # Create title for graph
        plt.legend() # Show the labels description for red and blue color
        plt.show() # Display the graph

    def predict(self, x):
        return np.squeeze(np.dot(x, self.weights) + self.bias) # flatten to 1D and predict the output

    def score(self, x, y):
        y = np.ravel(y)
        y_pred = self.predict(x)
        return self.r_squared(y_pred, y)

    def root_mean_squared_error(self, y_pred, y):
        y = np.ravel(y) # Convert y (actual value) from 2D into 1D
        y_pred = np.ravel(y_pred) # Convert y_pred (predicted value) from 2D to 1D
        return np.sqrt(np.mean((y_pred - y) ** 2))

    def r_squared(self, y_pred, y):
        y = np.ravel(y) # Convert y (actual value) from 2D into 1D
        y_pred = np.ravel(y_pred) # Convert y_pred (predicted value) from 2D to 1D

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
