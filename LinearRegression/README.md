# Linear Regression
**Linear Regression is used in continuous value prediction, such as house prices, and other stuff**\
In this example(project), we'll be using house feature & prices dataset, and the formula for linear regression

**Additional notes**\
    Difference between ravel and flatten in numpy\
    The difference between np.ravel() and np.flatten() lies in how its memory works and its impact towards original array

    1. np.ravel()
    - np.ravel() works by returning a view of the original array, meaning it does not create a copy of the original array
    - np.ravel() will override the original array, meaning it will modify the original array as well as they share the same memory address

    2. np.flatten()
    - np.flatten() works by returning a new copy of array from the original array
    - np.flatten() will not override the original array, as it is a new independent array from the original one and they share different memory addresses

# Custom Linear Regression Usage:
**Model Parameters:**\
`LinearRegressionGD(lr=0.001, regularization='lasso', alpha=0.0, penalty_loss=0.001, epoch=3000, init='random')`
1. `lr`: Learning Rate of the model, it refers to how fast the model steps down during gradient descent.\
If the Learning Rate is too high, the gradient descent will oscillate around the local minima and not converge.\
If the Learning Rate is too low, the gradient descent will not converge in time(under the number of iterations)
2. `regularization`: Represents the regularization techniques mentioned above, which are L1 Lasso, L2 Ridge and Elastic Net (L1+L2).\
`Params Choice: ('lasso', 'ridge', 'elastic_net')`
3. `alpha`: Represents the params for Elastic Net, where it is used to control the penalty for both L1 and L2
4. `penalty_loss`: Represents the regularisation parameter used in L1, L2 and Elastic Net
5. `epoch`: Represents the iterations performed by the Linear Regression model
6. `init`: Represents the initialization type towards initializing the weights and bias\
`Params Choice: ('zeros', 'random')`\
a) zeros: Initialize both weights and bias as matrices filled with 0\
b) random: Initialize both weights and bias as matrices filled with random numbers from 0 to 1\
`Random` will be preferred as it is faster to fit weights and bias from an existing random number than 0.

# Formula Breakdown:
```math
y = \theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n} + b
```
**Where:**\
y = Actual value\
x = Value for each feature\
$`\theta_{n}`$ = Weights for each feature (1-n)\
b = Bias

# Loss Formula: Mean Square Error
**Explanation:**\
- In continuous regression model, we'll be using Mean Square Error (MSE), as it takes the square of the difference between actual value and predicted value.
- This means if the difference (loss) is high, the penalty will be higher as the value is squared.
- Additionally, we have divided the total loss with the total number of dataset (n) to calculate the average loss. This is to prevent gradient exploding due to large loss value.

**Formula:**\
```math
\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}
```
**Where:**\
n = Number of total rows (Total dataset count)\
$`\hat{y}`$ = Predicted value\
y = Actual value

# Mean Square Error with L1 (Lasso) Regularisation
**Explanation:**
- In order to reduce the risk of overfitting, where our model memorise the data during training and under-perform during testing, we will be implement regularisation as well, which is L1 (Lasso)
- The purpose of L1 (Lasso) Regularisation is to add a penalty into the model, where a small difference in loss is further amplified into a huge value, on top on our loss function
- The penalty of L1 Lasso Regularisation is calculated by combining the sum of absolute value weights, as shown below:

**Formula:**\
```math
\lambda\sum_{i=1}^{m}|\theta_{i}|
```
**Where:**\
$`\lambda`$ = Regularisation penalty constant (recommended: 0.0001)\
m = Number of total columns (Total features in a dataset)\
$`\theta_{i}`$ = Weights for each feature (from 1 - m)\

**Combining L1 (Lasso) Regularisation with MSE:**\
```math
\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}|\theta_{i}|
```

# Mean Square Error with L2 (Ridge) Regularisation
**Explanation:**
- In order to reduce the risk of overfitting, where our model memorise the data during training and under-perform during testing, we will be implement regularisation as well, which is L2 (Ridge)
- The purpose of L2 (Ridge) Regularisation is to add a penalty into the model, where a small difference in loss is further amplified into a huge value, on top on our loss function
- The penalty of L2 Ridge Regularisation is calculated by combining the sum of squares of the weights, as shown below:

**Formula:**\
```math
\lambda\sum_{i=1}^{m}\theta_{i}^{2}
```

**Where:**\
$`\lambda`$ = Regularisation penalty constant (recommended: 0.0001)\
m = Number of total columns (Total features in a dataset)\
$`\theta_{i}`$ = Weights for each feature (from 1 - m)\

**Combining L2 (Ridge) Regularisation with MSE:**\
```math
\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}(\theta_{i})^{2}
```

# Mean Square Error with Elastic Net Regularisation
**Explanation:**
- In order to reduce the risk of overfitting, where our model memorise the data during training and under-perform during testing, we will be implement regularisation as well, which is Elastic Net (L1 + L2)
- The purpose of Elastic Net Regularisation is to combine the penalty from L1 Lasso and L2 Ridge, where it maximizes the benefits from both regularisation techniques, making it usually the most efficient regularisation technique
- The penalty of Elastic Net Regularisation is calculated by combining both L1 and L2 penalty, and introducing an alpha parameters to control both penalty value for bias-variance tradeoffs :

**Formula:**\
```math
\lambda(\alpha\sum_{i=1}^{m}|\theta_{i}| + (1-\alpha)\sum_{i=1}^{m}\theta_{i}^{2})
```

**Where:**\
$`\lambda`$ = Regularisation penalty constant (recommended: 0.0001)\
$`\alpha`$ = Alpha constant to control L1 and L2 penalty (recommended: 0.05)\
m = Number of total columns (Total features in a dataset)\
$`\theta_{i}`$ = Weights for each feature (from 1 - m)

**Combining L2 (Ridge) Regularisation with MSE:**\
```math
\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda(\alpha\sum_{i=1}^{m}|\theta_{i}| + (1-\alpha)\sum_{i=1}^{m}\theta_{i}^{2})
```

# Derivative of loss w.r.t Weights
```math
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}L(\theta)\\
& =\frac{\partial }{\partial \theta_{j}}(\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}\\
& =\frac{\partial }{\partial \theta_{j}}(\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2})\\
& =\frac{1}{2n}\sum_{i=1}^{n}\frac{\partial }{\partial \theta_{j}}(\hat{y_{i}}-y_{i})^{2}\\
& =\frac{1}{2n}\sum_{i=1}^{n}\frac{\partial }{\partial \theta_{j}}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))^{2}\\
& =\frac{2}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot \frac{\partial }{\partial \theta_{j}}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))\\
& =\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot ((0+0))-(x_{ij}+0))\\
& =\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})x_{ij}
\end{aligned}
```

**Where:**\
$`L(\theta)`$ = Loss function\
n = Number of total rows (Total dataset count)\
m = Number of total columns (Total dataset features)\
$`\frac{1}{n}\sum_{i=1}^{n}`$ = Sum of total rows (i from 1 to n)\
$`\frac{1}{n}\sum_{j=1}^{m}`$ = Sum of total columns (i from 1 to m)\
$`\hat{y}`$ = Predicted value\
y = Actual value\
$`\lambda`$ = L1 constant\
$`|\theta_{i}|`$ = Absolute value of weight with index i (i from feature 0 to m)

# Derivative of loss w.r.t Bias
```math
\begin{aligned}
& \frac{\partial }{\partial b}L(\theta)\\
& =\frac{\partial }{\partial b}(\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}\\
& =\frac{\partial }{\partial b}(\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2})\\
& =\frac{1}{2n}\sum_{i=1}^{n}\frac{\partial }{\partial b}(\hat{y_{i}}-y_{i})^{2}\\
& =\frac{1}{2n}\sum_{i=1}^{n}\frac{\partial }{\partial b}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))^{2}\\
& =\frac{2}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot \frac{\partial }{\partial b}((\hat{\theta_{i}x_{i}+b})-(\theta_{i}x_{i}+b))\\
& =\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})\cdot ((0+0))-(0+0))\\
& =\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})
\end{aligned}
```

**Where:**\
$`L(\theta)`$ = Loss function\
n = Number of total rows (Total dataset count)\
m = Number of total columns (Total dataset features)\
$`\frac{1}{n}\sum_{i=1}^{n}`$ = Sum of total rows (i from 1 to n)\
$`\frac{1}{n}\sum_{j=1}^{m}`$ = Sum of total columns (i from 1 to m)\
$`\hat{y}`$ = Predicted value\
y = Actual value\
$`\lambda`$ = L1 constant\
$`|\theta_{i}|`$ = Absolute value of weight with index i (i from feature 0 to m)\

# Derivative of L1 Lasso Regularisation
```math
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}(\lambda\sum_{i=1}^{m}|\theta_{i}|)\\
&= \lambda\cdot  \begin{cases} -1& \text{if } \theta_{j} < 0 \\ {[-1, 1]} & \text{if } \theta_{j} = 0 \\ 1& \text{if } \theta_{j}> 0 \end{cases}\\
&= \lambda\cdot \text{sign}(\theta_{j}), \text{sign}(0)\in [-1, 1]
\end{aligned}
```

**Where:**\
$`\lambda`$ = Regularisation penalty constant\
$`\theta_{j}`$ = Weights at the jth feature\
sign($`\theta_{j}`$) = Make $`\theta`$ as -1 if negative, 1 if positive and 0 if = 0 since it is not differentiable at 0

# Derivative of L2 Ridge Regularisation
```math
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}(\frac{\lambda}{2}\sum_{i=1}^{m}(\theta_{i})^{2})\\
&= \frac{\lambda}{2}\cdot 2\cdot \theta_{j}\\
&= \lambda\theta_{j}
\end{aligned}
```

**Where:**\
$`\lambda`$ = Regularisation penalty constant\
$`\theta_{j}`$ = Weights at the jth feature

# Derivative of Elastic Net Regularisation
```math
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}[\lambda(\alpha\sum_{i=1}^{m}|\theta_{i}| + (1-\alpha)\sum_{i=1}^{m}(\theta_{i})^{2})]\\
&= \lambda\alpha\begin{cases} -1& \text{if } \theta_{j} < 0 \\ {[-1, 1]} & \text{if } \theta_{j} = 0 \\ 1& \text{if } \theta_{j}> 0 \end{cases} + 2\lambda(1-\alpha)\theta_{j}\\
&= \lambda\alpha\cdot \text{sign}(\theta_{j})+2\lambda(1-\alpha)\theta_{j}, \text{sign(0)}\in[-1, 1]
\end{aligned}
```

**Where:**\
$`\lambda`$ = Regularisation penalty constant\
$`\alpha`$ = Mixing Ratio for L1 and L2\
$`\theta_{j}`$ = Weights at the jth feature\
sign($`\theta_{j}`$) = Make $`\theta`$ as -1 if negative, 1 if positive and 0 if = 0 since it is not differentiable at 0

# Root Mean Square Error:
It is in short, the square root of the mean square error function we have explained earlier:\

**Formula:**
```math
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}}
```

# R-Square Formula
- It is used to calculate how well our linear regression model fits with the dataset. Think of it as an accuracy score for Linear Regression itself.
- We do not use accuracy score to calculate the accuracy of our linear regression model as it is impossible for our model to predict the exact same value as the actual value (i.e. \$`4700000 vs \`$4700000). Thus, we estimate how close our model is in guessing the actual value correctly.
- In R-Square score, it ranges from 0 to 1 where 0 indicates the model is just random guessing while 1 is a perfect fit. The lower the loss value, the higher the R-Square value.

**Formula:**
```math
1 - \frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^{2}}
```

**Where:**\
$`y_{i}`$ = Actual value for ith index\
$`\hat{y}_{i}`$ = Predicted value for ith index\
$`\bar{y}_{i}`$ = Mean of the actual value
