import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr, l2, epoch):
        self.lr = lr
        self.l2 = l2
        self.epoch = epoch
        self.weights = None
        self.bias = None
        self.epsilon = 1e-8

    def sigmoid(self, x):
        # This will be the forward propagation for our logistic regression
        # The reason we need activation function like sigmoid is to introduce non-Linearly relationship
        # In simple terms, it is a logic gate switch for machine learning models
        # Without it, your model will be a normal regression instead of a classification
        return 1 / (1 + np.exp(-x))

    # You may use oneHotEncode for multi-class classification, but since we're doing binary classification this is not needed
    # def oneHotEncode(self, y, k): # The goal of oneHotEncoding is to expand the array from 1D to n dimensional, with n being the number of classes
    #     y_idx = y.reshape(-1) # So this one flatten the y array from (303, 1) to (303, ) 1D
    #     m = y_idx.shape[0] # We get the row number from y_idx array which is 303
    #
    #     onehotencoded_y = np.zeros((m, k)) # So this one we expand the y array into (m, k), where m is the number of records(rows) and k is the number of class
    #                                        # Since we pass in the class number k as parameter, it helps smoothen the process
    #                                        # So we'll have (303, 2) in final, with the column 0 as yes and column 1 as no
    #     onehotencoded_y[np.arange(m), y_idx] = 1 # In np.arange(m), we start the record from 0 to 302, and for each record we match it with its value in y_idx array
    #                                              # For example, if the first record has no heart disease, being 1, then [0, 1] will be value of 1
    #     # In essence here's the visualization of oneHotEncoding
    #     # Before: [0, 1, 0, 1, 0, 1,...]
    #     '''After:
    #     [[0, 1],
    #     [1, 0],
    #     [0, 1],
    #     [1, 0],
    #     [0, 1],
    #     [1, 0],...]
    #     '''
    #     return onehotencoded_y

    def binary_cross_entropy_loss(self, y, z):
        y = np.asarray(y).reshape(-1, 1)
        z = np.asarray(z).reshape(-1, 1)
        m = y.shape[0] # Get the number of records (rows) of y
        return -1/m * np.sum(y * np.log(z + self.epsilon) + (1 - y) * np.log(1 - z + self.epsilon)) # This follows the formula of BCE, where epsilon is added to prevent log(0) error from happening

    def fit(self, X_train, y_train, X_test, y_test):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1, 1)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test).reshape(-1, 1)

        m, n = X_train.shape # Initialize the attributes for row (m) and column(n) for training X
        # k = int(np.max(y_train_idx)) + 1 # Go through the row value in y_train, if the highest value is 0, meaning there's only 0+1 = 1 class. If the highest is 1, meaning there's 1+1 = 2 classes

        # Initialize weights and biases
        self.weights = np.random.randn(n, 1) / np.sqrt(n) # Uniform random initialization of weight with shape of (n, 1)
        self.bias = np.random.randn(1, 1) # Random initialization of bias with shape of (1, 1)

        # Declaring losses and accuracies
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        # Math time
        for _ in range(self.epoch):
            train_logits = np.dot(X_train, self.weights) + self.bias # Normal y = mX + c, shape (m, 1)
            z_train = self.sigmoid(train_logits) # Activation function with sigmoid to get z, shape (m, 1)
            train_loss = self.binary_cross_entropy_loss(y_train, z_train) # Calculate training loss using BCE
            train_loss += self.l2 * np.sum(np.square(self.weights)) # Add training loss together with l2 penalty
            train_losses.append(train_loss) # Add the training loss for ith epoch into the train_losses array

            test_logits = np.dot(X_test, self.weights) + self.bias # Normal y = mX + c, shape (m, 1)
            z_test = self.sigmoid(test_logits) # Activation function with sigmoid to get z, shape (m, 1)
            test_loss = self.binary_cross_entropy_loss(y_test, z_test) # Calculate testing loss using BCE. No Ridge regularization as this is testing phase, strictly only evaluate the results with no tweaks
            test_losses.append(test_loss) # Add the testing loss for the ith epoch into the test_losses array

            # Mini backpropagation and gradient descent
            gradient = 1 / m * (z_train - y_train) # Formula = 1/m (y_hat - y), where you normalize or get the average by dividing it with the number of records; shape = (m, 1)
            dW = np.dot(X_train.T, gradient) + 2 * self.l2 * self.weights # Derivative for weights, X_train.T (n, m) @ gradient (m, 1) shape (n, 1)
            db = np.sum(gradient, axis=0, keepdims=True) # Derivative for biases, shape = (1, 1)

            # Gradient Descent
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            if _ % 100 == 0:
                train_pred = (z_train >= 0.5).astype(int) # Predict by rounding off the probability to either 0 or 1
                train_acc = float(np.mean(train_pred == y_train)) # Match the prediction with results and calculate the accuracy
                train_accuracies.append(train_acc) # Add the train accuracy result into train_accuracies array

                test_pred = (z_test >= 0.5).astype(int) # Predict by rounding off the probability to either 0 or 1
                test_acc = float(np.mean(test_pred == y_test)) # Match the prediction with results and calculate the accuracy
                test_accuracies.append(test_acc) # Add the test accuracy result into test_accuracies array

                print(f"Epoch {_}: Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

        plt.plot(range(self.epoch), train_losses, label="Training Loss", color="red")
        plt.plot(range(self.epoch), test_losses, label="Test Loss", color="blue", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.show()

        return self

    def predict_proba(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias) # Calculate the probability based on prediction result with sigmoid

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int) # Get the probability, if > 0.5 then convert to integer and get the value

