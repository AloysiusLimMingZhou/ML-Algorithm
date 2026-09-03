# Chapter 2.1: Logistic Regression

As you may have learned Linear Regression & basics of regularization involving L2 Ridge in `Chapter 1: Linear Regression`, we'll be moving on to the next task, which is `Classification Task`

## Recall in Previous Chapter
In previous chapter, we have learned how to train a model to learn and predict continuous value (`Gaussian Distribution`) using `Linear Regression`. In this chapter, we'll be looking into a different data distribution, which is `Bernoulli Distribution`

## Comparison between Gaussian and Bernoulli Distribution
![Gaussian Distribution](/LogisticRegressionImage/Gaussian-Distribution.png)\
**Gaussian Distribution Graph**

- In Gaussian Distribution, it refers to a continuous data distribution with real values data. Basically data that has actual number and not just probability values (i.e. 400000 vs 0.34583)
- Exp: House pricing, where the house prices are represented in normal number values with infinite possibilities instead of probabilistic
- To simplify, any data that has continuous value (300000, 12354, -2, 9382.302,...) will most likely fall under Gaussian, while probabilistic & discrete values (0.554, 0.345, 0.0001, 1, 0) will have a distribution of Bernoulli 

![Bernoulli Distribution](/LogisticRegressionImage/Bernoulli-Distribution.webp)\
**Bernoulli Distribution Graph**

- In Bernoulli Distribution, it refers to discrete data where it records the probability of the outcome occurs (i.e. Chances of landing head on a coin flip is 0.554)
- It is also known as classification task, where you predict the outcome of the result given input data (0 for false, 1 for true)
- Exp: Coin flip (0 for head, 1 for tail)

Let's take a look at some classic dataset with continuous values and discrete values
## Dataset with Continuous Value
- House Pricing Data

## Dataset with Discrete Value
- Email Spam Classification (Spam/Not Spam)
- Tumour cell Classification (Malignant/Benign)

## How to relate Linear Regression to Logistics Regression

- In linear regression, our goal is to simply draw the best fit line on the data plot that minimizes the loss to get an accurate prediction
- On the other hand, in logistics regression, similarly in linear regression, we will first calculate the predicted output, $\hat{y} = X\theta + b$, but we will add an extra step by passing this predicted output into a function, which is the `Sigmoid Function`
- The goal of sigmoid function is to convert the continuous predicted value we have earlier into discrete probabilistic value which ranges from 0 to 1.
- The converted values we will call it as **logits**, and it helps us to predict & classify each data row where logits > 0.5 will be classified as 1 while logits < 0.5 will be classified as 0.

Exp: Converting Continuous Value into discrete, then classify them based on 0.5 threshold
$$
X\theta + b = \begin{bmatrix}1.45\\ 2.56\\ -1.78\\ -4.048\end{bmatrix} = \begin{bmatrix}0.81\\ 0.93\\ 0.144\\ 0.017\end{bmatrix} = \begin{bmatrix}1\\1\\0\\0\end{bmatrix}
$$

## Sigmoid function explanation
![Sigmoid Function](/LogisticRegressionImage/Sigmoid-Activation-Function.png)

**Formula:**
$$
\begin{aligned}
z = X\theta + b\\
\sigma(x) = \frac{1}{1+e^{-z}}
\end{aligned}
$$

Where:\
$\sigma(x)$: Sigmoid function\
e: Exponential value

- Notice that we swap the notation from $\hat{y}$ to $z$, this is because we usually use $\hat{y}$ as our final predicted output, and since that is not the case in logistics, we will just use z for a placeholder notation instead.
- As you can see from the graph here, by using sigmoid function, it forces our continuous predicted output into probabilistic(discrete) values as its function range is only [0, 1]
- This means that output after passing through sigmoid can only be between 0 and 1, which satisfy probability concepts as well, since values cannot be more than 1 and less than 0.

Thus, Logistics Regression essentially is just Linear Regression with extra steps to convert continuous numbers into discrete probabilistic numbers, then in the end classify it to be either binary 0 or 1.

## Steps in Logistics Regression:
1. Calculate the output by $z = X\theta$ + b
2. Pass in the output with sigmoid activation function to form probabilistic logits ($\frac{1}{1+e^{-z}}$)
3. Pass the logits into the loss with cost function (Binary Cross Entropy)
4. Fit (minimize) the gradient of the loss by adjusting the weights and bias values using gradient descent
5. Repeat this for epoch number of times(i.e. 2000) until it converges (meaning the gradient calculated is already minimized)
6. To use the trained Logistics Regression to predict values, create a threshold where if the predicted logits is greater than 0.5, it'll be classified as class 1. Otherwise, it'll be classified as class 0.

## Binary Cross Entropy Loss
- Unlike in `Linear Regression` where we use `Mean Square Error`, it is not applicable in Logistic Regression, and we'll be using another loss function to minimize the loss of the model, which is called `Binary Cross Entropy Loss`
- The motivation behind using different loss functions and why the specific formula can be further derived from Maximum Likelihood Estimation (MLE), which will be covered in future chapters. For now just trust the formula

**Formula:**
**Summation Form:**
$$
L(\theta) = -\frac{1}{n}\sum_{i=1}^{n}(y_i\ln(\hat{y_i}) + (1-y_i)\ln(1-\hat{y_i}))
$$

Where:\
n = Total number of data in datasets (total row)\
$\hat{y}_i$: Predicted output for ith data row (The probabilistic logits)\
ln: Natural logarithm

**Vector Form:**
$$L(\theta) = \frac{1}{n}(Y\cdot \ln(\hat{Y}) + (1-Y)\cdot \ln(1-\hat{Y}))$$

To think of how the loss function works:
- It takes in the predicted $\hat{y}$ which is the logits (i.e. 0.667) and compare it with the actual classified value y (i.e. 1)
- If the predicted probabilistic value is closer to the actual classification (0.667 is almost close to 1) meaning the model is getting more accurate, the loss given will become lower since the model is accurate
- If the predicted probabilistic value is farther from the actual classification (0.667 is very far from 0), then the model is wrong and larger loss will be given to tell the model to adjust the wrong prediction
- The weights and bias will then be adjusted based on the loss using gradient descent to ensure it predicts all data accurately by their actual class (1 or 0):

## Gradient Descent
**Weights:** $\theta_{j}=\theta_{j}-\alpha\frac{\partial }{\partial \theta_{j}}L(\theta)$\
**Bias:** $b=b-\alpha\frac{\partial }{\partial b}L(\theta)$

**Where:**\
$L(\theta)$ = Cost Function / Loss\
$\alpha$ = Learning Rate\
$\theta_{j}$ = Weights for the jth column (jth feature)\
b = Bias

## Binary Cross Entropy Loss with L2 Ridge
- If your model is slightly overfitting (where training performs better than testing i.e. 90% accuracy vs 30% accuracy), you can try to add L2 Ridge into the loss.
- Recall in previous chapter `Chapter 1.3: Multicollinearity`, if the model is too complex with many types of features, you can add L2 Ridge to simplify the model, as its added penalty into the loss will cause irrelevant weights to become smaller values close to 0, which effectively reduce its influence as random noise to the model.

**Formula:**
**Summation Form:**
$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot ln(\hat{y}_{i})+(1-y_{i})\cdot ln(1-\hat{y}_{i}))+\lambda\sum_{k=1}^{m}(\theta^{2}_{k})
$$

**Where:**\
$\theta$ = Weights with respect to Cost Function(loss)\
n = Total rows of data\
ln = Natural logarithm\
$y_{i}$ = Actual output for ith data row (0 or 1)\
$\hat{y}_{i}$ = Predicted output for ith data row (0 or 1)\
$\lambda$ = L2 Ridge Regularization constant
Penalty = $\lambda\sum_{i=1}^{n}(\theta^{2})$

**Vector Form:**
$$L(\theta) = \frac{1}{n}(Y\cdot \ln(\hat{Y}) + (1-Y)\cdot \ln(1-\hat{Y}) + \lambda\theta^2)$$

## Derivation of Loss in Gradient Descent
- Since we are using gradient descent to solve our logistic Regression problem, we'll have to compute the derivative of our loss with respect to weight and bias

**Derivative of Cost Function with respect to weights:**\
**Summation Form:**
$$
\begin{aligned} 
& \frac{\partial }{\partial \theta_{j}} L(\theta)\\ 
&=\frac{\partial }{\partial \theta_{j}}[(-\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot ln(\hat{y}_{i})+(1-y_{i})\cdot ln(1-\hat{y}_{i})))]\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot ln(\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}})+(1-y_{i})\cdot ln(1-\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}})])\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot ln(\frac{1}{1+e^{-z_{i}}})+(1-y_{i})\cdot ln(1-\frac{1}{1+e^{-z_{i}}})])\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial \theta_{j}}[y_{i}\cdot ln(g(z_{i}))+(1-y_{i})\cdot ln(1-g(z_{i}))])\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}\cdot \frac{1}{g(z_{i})}+(1-y_{i})\cdot \frac{1}{1-g(z_{i})}\cdot \frac{\partial }{\partial \theta_{j}}g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial \theta_{j}}z_{i})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial \theta_{j}}\sum_{k=1}^{m}(x_{ik}\theta_{k}+b))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})})\cdot g(z_{i})(1-g(z_{i}))\cdot x_{ij})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}((-y_{i}(1-g(z_{i}))+(1-y_{i})g(z_{i}))\cdot x_{ij})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}+y_{i}g(z_{i})+g(z_{i})-y_{i}g(z_{i}))x_{ij}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(g(z_{i})-y_{i})x_{ij}\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})x_{ij}\\ 
\end{aligned}
$$
**Vector Form:**
$$\frac{\partial }{\partial \theta_{j}} L(\theta) = \frac{1}{n}X^T(\hat{Y} - Y)$$


**Derivative of Cost Function with respect to bias:**\
**Summation Form**
$$
\begin{aligned} 
& \frac{\partial }{\partial b} L(\theta)\\ 
&=\frac{\partial }{\partial b}(-\frac{1}{n}\sum_{i=1}^{n}(y_{i}\cdot ln(\hat{y}_{i})+(1-y_{i})\cdot ln(1-\hat{y}_{i}))+\lambda\sum_{k=1}^{m}(\theta^{2}_{k}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot ln(\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}})+(1-y_{i})\cdot ln(1-\frac{1}{1+e^{-\sum_{k=1}^{m}(x_{ik}\theta_{k}+b)}}))+0)\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot ln(\frac{1}{1+e^{-z_{i}}})+(1-y_{i})\cdot ln(1-\frac{1}{1+e^{-z_{i}}})))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-\frac{\partial }{\partial b}(y_{i}\cdot ln(g(z_{i}))+(1-y_{i})\cdot ln(1-g(z_{i}))))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}\cdot \frac{1}{g(z_{i})}\cdot \frac{\partial }{\partial b}g(z_{i})+(1-y_{i})\cdot \frac{1}{1-g(z_{i})}\cdot \frac{\partial }{\partial b}g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial b}z_{i})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot \frac{\partial }{\partial b}\sum_{k=1}^{m}(x_{ik}\theta_{k}+b))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}([-y_{i}\frac{1}{g(z_{i})}+(1-y_{i})\frac{1}{1-g(z_{i})}]\cdot g(z_{i})(1-g(z_{i}))\cdot 1)\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}(1-g(z_{i}))+(1-y_{i})g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(-y_{i}+y_{i}g(z_{i})+g(z_{i})-y_{i}g(z_{i}))\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(g(z_{i})-y_{i})\\ 
&=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})\\ 
\end{aligned}
$$

**Where:**\
$L(\theta)$ = Cost Function / Loss\
$\theta_{j}$ = Weights for the jth column (from j to m of features)\
n = Total rows of data (from i to n of data)\
$y_{i}$ = Actual output for ith data row (0 or 1)\
$\hat{y}_{i}$ = Predicted output for ith data row (0 or 1)\
Derivative = $\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})$\
$x_{ij}$ = Entire X_train or X_test (contains from i to n of total data and from j to m of total features)\
$\lambda$ = Regularization penalty constant\
Penalty = $\lambda\sum_{k=1}^{m}(\theta^{2}_{k})$, where it takes the square of each weights from kth = 1 feature to mth feature

**Vector Form**
$$\frac{\partial }{\partial \theta_{j}} L(\theta) = \frac{1}{n}(\hat{Y} - Y)$$

# Derivative of L2 Ridge Regularisation
**Summation Form**:
$$
\begin{aligned}
& \frac{\partial }{\partial \theta_{j}}(\lambda\sum_{i=1}^{m}(\theta_{i})^{2})\\
&= \lambda\cdot 2\cdot \theta_{j}\\
&= 2\lambda\theta_{j}
\end{aligned}
$$

**Where:**\
$\lambda$ = Regularisation penalty constant\
$\theta_{j}$ = Weights at the jth feature\
$\theta$ = Vector Weights (Shape: m, 1)

**Vector Form:**
$$\frac{\partial }{\partial \theta}(\lambda\theta^{2}) = 2\lambda\theta$$


# Final Loss Derivative wrt different regularization:
**2. BCE and L2**:\
**Summation Form:**
$$
\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})x_{ij} + 2\lambda\theta_{j}
$$

**Vector Form:**
$$\frac{1}{n}X^T(\hat{Y}-Y) + 2\lambda\theta$$