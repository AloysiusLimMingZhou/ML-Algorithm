# Linear Regression
**Linear Regression is used in continuous value prediction, such as house prices, and other stuff**\
In this example(project), we'll be using house feature & prices dataset, and the formula for linear regression

**Additional notes**\
    Difference between ravel and flatten in numpy
    The difference between np.ravel() and np.flatten() lies in how its memory works and its impact towards original array

    1. np.ravel()
    - np.ravel() works by returning a view of the original array, meaning it does not create a copy of the original array
    - np.ravel() will override the original array, meaning it will modify the original array as well as they share the same memory address

    2. np.flatten()
    - np.flatten() works by returning a new copy of array from the original array
    - np.flatten() will not override the original array, as it is a new independent array from the original one and they share different memory addresses

# Formula Breakdown:
```math
y = \theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n} + b
```
**Where:**\
y = Actual value\
x = Value for each feature\
$\theta_{n}$ = Weights for each feature (1-n)\
b = Bias

# Loss Formula: Mean Square Error
**Explanation:**\
- In continuous regression model, we'll be using Mean Square Error (MSE), as it takes the square of the difference between actual value and predicted value.
- This means if the difference (loss) is high, the penalty will be higher as the value is squared.
- Additionally, we have divided the total loss with the total number of dataset (n) to calculate the average loss. This is to prevent gradient exploding due to large loss value.

**Formula:**\
```math
\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}
```
**Where:**\
n = Number of total rows (Total dataset count)\
$\hat{y}$ = Predicted value\
y = Actual value

# Mean Square Error with L1 (Lasso) Regularisation
**Explanation:**
- In order to reduce the risk of overfitting, where our model memorise the data during training and under-perform during testing, we will be implement regularisation as well, which is L1 (Lasso)
- The purpose of L1 (Lasso) Regularisation is to add a penalty into the model, where a small difference in loss is further amplified into a huge value, on top on our loss function

**Formula:**\
```math
\lambda\sum_{i=1}^{m}|\theta_{i}|
```
**Where:**\
$\lambda$ = l1 penalty constant (recommended: 0.0001)\
m = Number of total columns (Total features in a dataset)\
$w_{i}$ = Weights for each feature (from 1 - m)\

**Combining L1 (Lasso) Regularisation with MSE:**\
```math
\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}|\theta_{i}|
```

# Derivative of loss w.r.t Weights
```math
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}L(\theta)\\
& =\frac{\partial }{\partial \theta_{j}}(\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}|\theta_{i}|)\\
& =\frac{\partial }{\partial \theta_{j}}(\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2})+\frac{\partial }{\partial \theta_{j}}(\lambda\sum_{i=1}^{m}|\theta_{i}|)\\
& =\frac{1}{n}\sum_{i=1}^{n}\frac{\partial }{\partial \theta_{j}}(\hat{y_{i}}-y_{i})^{2}+\lambda\sum_{i=1}^{m}\frac{\partial }{\partial \theta_{j}}(|\theta_{i}|)\\
\end{aligned}
```
```math
\begin{aligned}
& =\frac{1}{n}\sum_{i=1}^{n}\frac{\partial }{\partial \theta_{j}}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))^{2}+\lambda\sum_{i=1}^{m}\frac{\partial }{\partial \theta_{j}}(|\theta_{i}|)\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot \frac{\partial }{\partial \theta_{j}}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))+\lambda\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot ((0+0))-(x_{ij}+0))+\lambda\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})x_{ij}+\lambda
\end{aligned}
```

**Where:**\
$L(\theta)$ = Loss function\
n = Number of total rows (Total dataset count)\
m = Number of total columns (Total dataset features)\
$\frac{1}{n}\sum_{i=1}^{n}$ = Sum of total rows (i from 1 to n)\
$\frac{1}{n}\sum_{j=1}^{m}$ = Sum of total columns (i from 1 to m)\
$\hat{y}$ = Predicted value\
y = Actual value\
$\lambda$ = L1 constant\
$|\theta_{i}|$ = Absolute value of weight with index i (i from feature 0 to m)

# Derivative of loss w.r.t Bias
```math
\begin{aligned}
& \frac{\partial }{\partial b}L(\theta)\\
& =\frac{\partial }{\partial b}(\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}|\theta_{i}|)\\
& =\frac{\partial }{\partial b}(\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2})+\frac{\partial }{\partial b}(\lambda\sum_{i=1}^{m}|\theta_{i}|)\\
& =\frac{1}{n}\sum_{i=1}^{n}\frac{\partial }{\partial b}(\hat{y_{i}}-y_{i})^{2}+\lambda\sum_{i=1}^{m}\frac{\partial }{\partial b}(|\theta_{i}|)\\
& =\frac{1}{n}\sum_{i=1}^{n}\frac{\partial }{\partial b}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))^{2}\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot \frac{\partial }{\partial b}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot ((0+0))-(0+0))\\
& =\frac{2}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})
\end{aligned}
```

**Where:**\
$L(\theta)$ = Loss function\
n = Number of total rows (Total dataset count)\
m = Number of total columns (Total dataset features)\
$\frac{1}{n}\sum_{i=1}^{n}$ = Sum of total rows (i from 1 to n)\
$\frac{1}{n}\sum_{j=1}^{m}$ = Sum of total columns (i from 1 to m)\
$\hat{y}$ = Predicted value\
y = Actual value\
$\lambda$ = L1 constant\
|$\theta_{i}$| = Absolute value of weight with index i (i from feature 0 to m)\

# Root Mean Square Error:
It is in short, the square root of the mean square error function we have explained earlier:\

**Formula:**
```math
\sqrt{\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}}
```

# R-Square Formula
- It is used to calculate how well our linear regression model fits with the dataset. Think of it as an accuracy score for Linear Regression itself.
- We do not use accuracy score to calculate the accuracy of our linear regression model as it is impossible for our model to predict the exact same value as the actual value (i.e. \$4700000 vs \$4700000). Thus, we estimate how close our model is in guessing the actual value correctly.
- In R-Square score, it ranges from 0 to 1 where 0 indicates the model is just random guessing while 1 is a perfect fit. The lower the loss value, the higher the R-Square value.

**Formula:**
```math
1 - \frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^{2}}
```

**Where:**\
$y_{i}$ = Actual value for ith index\
$\hat{y}\_{i}$ = Predicted value for ith index\
$\bar{y}\_{i}$ = Mean of the actual value


