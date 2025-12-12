# Logistic Regression Formula Breakdown
**Aim: To use a similar approach like Linear Regression but to calculate the probability of an event most likely to occur**

# Formula: 
$$\frac{1}{1+e^{-(b_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{m}x_{m})}}\= \frac{1}{1+e^{-z}}$$

**Where:**\
$\theta_{m}$ = Weights for mth column (mth feature)\
$x_{m}$ = Value for the mth column (mth feature)\
$b_{0}$ = Bias

**Sigmoid Function Breakdown:**\
$$g(z)=\frac{1}{1+e^{-z}}$$\
$$z=\sum_{i=1}^{n}x_{i}\theta_{i}+b$$

**Derivative of sigmoid function in logistic regression:**\
```math
\begin{aligned}
\frac{\partial }{\partial z}g(z) = \frac{\partial }{\partial z}(\frac{1}{1+e^{-z}})\\=\frac{0-(-e^{-z})}{(1+e^{-z})^{2}}\\=\frac{e^{-z}}{(1+e^{-z})^{2}}\\=\frac{1}{1+e^{-z}}\cdot (1-\frac{1}{1+e^{-z}})\\=g(z)\cdot (1-g(z))
\end{aligned}
```
**Where:**\
z = $\sum_{i=1}^{n}x_{i}\theta_{i}+b$\
g(z) = Sigmoid Function

# Cost Function (With L2 Ridge Regularization):
The goal of the cost function or loss is to calculate how wrong our prediction is. Our goal is to minimize the cost function by adjusting our weights and biases.
The lower the loss, the higher the accuracy.

The use of L2 Ridge Regularization is to reduce the chances of model from overfitting. It is possible by adding a penalty that scales exponentially with the weights (weights^2).

The penalty helps the model to be more careful with loses as small difference in loss can result in large value of penalty.

**Formula:**\
$L(\theta)=\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot log(\hat{y}_{i})+(1-y_{i})\cdot log(1-\hat{y}_{i}))+\lambda\sum_{k=1}^{m}(\theta^{2}_{k})$\
**Where:**\
$\theta$ = Weights with respect to Cost Function(loss)\
n = Total rows of data\
log = Natural Logarithm\
$y_{i}$ = Actual output for ith data row (0 or 1)\
$\hat{y}_{i}$ = Predicted output for ith data row (0 or 1)\
$\lambda$ = L2 Ridge Regularization constant
Penalty = $\lambda\sum_{i=1}^{n}(\theta^{2})$

# Gradient Descent
In order to lower our loss until it reaches the minimum value, we need gradient descent to gradually reduce the loss by adjusting its gradient\
Thus, we have to find the derivative of our coss function, or loss, in order to minimize the loss such that its gradient = 0

**Derivative of Cost Function with respect to bias:**\
```math
\begin{aligned} 
& \frac{\partial }{\partial b} L(\theta)\\ 
&=\frac{\partial }{\partial b}(-\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot log(\hat{y}_{i})+(1-y_{i})\cdot log(1-\hat{y}_{i}))+\lambda\sum_{k=1}^{m}(\theta^{2}_{k}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot log(\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}})+(1-y_{i})\cdot log(1-\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}}))+0)\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot log(\frac{1}{1+e^{-z_{i}}})+(1-y_{i})\cdot log(1-\frac{1}{1+e^{-z_{i}}})))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot log(g(z_{i}))+(1-y_{i})\cdot log(1-g(z_{i}))))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}\cdot \frac{1}{g(z_{i})}\cdot \frac{\partial }{\partial b}g(z_{i})+(1-y_{i})\cdot \frac{1}{1-g(z_{i})}\cdot \frac{\partial }{\partial b}g(z_{i}))\\
\end{aligned}
```
```math
\begin{aligned}
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial b}z_{i})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial b}\sum_{k=1}^{m}(x_{ik}\theta_{k}+b))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot 1)\\
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}(1-g(z_{i}))+(1-y_{i})g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}+y_{i}g(z_{i})+g(z_{i})-y_{i}g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(g(z_{i})-y_{i})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})\\ 
\end{aligned}
```

**Derivative of Cost Function with respect to weights and Ridge L2 Regularisation:**\
```math
\begin{aligned} 
& \frac{\partial }{\partial \theta_{j}} [L(\theta)+\lambda\sum_{k=1}^{m}(\theta^{2}_{k})]\\ 
&=\frac{\partial }{\partial \theta_{j}}[(-\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot log(\hat{y}_{i})+(1-y_{i})\cdot log(1-\hat{y}_{i})))+\lambda\sum_{k=1}^{m}(\theta^{2}_{k})]\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot log(\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}})+(1-y_{i})\cdot log(1-\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}}))])+\frac{\partial }{\partial \theta_{j}}\lambda\sum_{k=1}^{m}(\theta^{2}_{k})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot log(\frac{1}{1+e^{-z_{i}}})+(1-y_{i})\cdot log(1-\frac{1}{1+e^{-z_{i}}}))])+\frac{\partial }{\partial \theta_{j}}\lambda\sum_{k=1}^{m}(\theta^{2}_{k})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot log(g(z_{i}))+(1-y_{i})\cdot log(1-g(z_{i}))])+\frac{\partial }{\partial \theta_{j}}\lambda\sum_{k=1}^{m}(\theta^{2}_{k})\\
\end{aligned}
```
```math
\begin{aligned}
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}\cdot \frac{1}{g(z_{i})}+(1-y_{i})\cdot \frac{1}{1-g(z_{i})}\cdot \frac{\partial }{\partial \theta_{j}}g(z_{i}))+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial \theta_{j}}z_{i})+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial \theta_{j}}\sum_{k=1}^{m}(x_{ik}\theta_{k}+b))+2\lambda\theta_{j}\\
\end{aligned}
```
```math
\begin{aligned}
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot x_{ij})+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}(1-g(z_{i}))+(1-y_{i})g(z_{i}))\cdot x_{ij})+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}+y_{i}g(z_{i})+g(z_{i})-y_{i}g(z_{i}))x_{ij}+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(g(z_{i})-y_{i})x_{ij}+2\lambda\theta_{j}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})x_{ij}+2\lambda\theta_{j}\\ 
\end{aligned}
```

**Where:**\
$L(\theta)$ = Cost Function / Loss\
$\theta_{j}$ = Weights for the jth column (from j to m of features)\
n = Total rows of data (from i to n of data)\
$y_{i}$ = Actual output for ith data row (0 or 1)\
$\hat{y}_{i}$ = Predicted output for ith data row (0 or 1)\
Derivative = $\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})$\
$x_{ij}$ = Entire X_train or X_test (contains from i to n of total data and from j to m of total features)\
$\lambda$ = L2 Ridge Regularization constant\
Penalty = $\lambda\sum_{k=1}^{m}(\theta^{2}_{k})$, where it takes the square of each weights from kth = 1 feature to mth feature

**Update Weights and Bias Formula:**\
**Weights:** $\theta_{j}=\theta_{j}-\alpha\frac{\partial }{\partial \theta_{j}}L(\theta)$\
**Bias:** $b=b-\alpha\frac{\partial }{\partial b}L(\theta)$

**Where:**\
$L(\theta)$ = Cost Function / Loss\
$\alpha$ = Learning Rate\
$\theta_{j}$ = Weights for the jth column (jth feature)\
b = Bias

# Step by step for Logistic Regression
1. Calculate the output by weights * X + bias
2. Pass in the output with sigmoid activation function (1/1+e^output)
3. Calculate the loss with cost function
4. Fit (minimize) the gradient of the loss by adjusting the weights and bias values using gradient descent
5. Repeat this for epoch number of times(i.e. 2000) until it converges (meaning the gradient calculated is already minimized)
