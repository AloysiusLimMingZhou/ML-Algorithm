# Chapter 4.5: Optimization Algorithms in Lasso Regression
- In this chapter, we'll be covering about:
1. Types of Optimization Algorithms for Lasso than Gradient Descent
2. Why Gradient Descent is not encouraged and inaccurate for Lasso Regularization optimization
3. Difference between optimization algorithms and when to use which

**Optimization Algorithms for Lasso Regression**
- Recall in `Chapter 1.2: Advanced topics in Linear Regression`, we have briefly talked about the 3 optimizations for Lasso, which are:
1. `Coordinate Descent`
2. `Proximal Gradient Descent with Soft Tresholding`
3. `Least Angle Regression (LARS)`
- In this part we will be covering each of them in detail

**Pre-requisite notations**
Below are the common math formulas or notations that you'll see in many statistical math notes or books:
1. $min_x f(x)$: min refers to minimizing the value of the function output, f(x). 
    This is commonly found in loss function like MSE in Linear Regression, where $min_\theta \frac{1}{2n}(y-X\theta)^2$. This is because in Linear Regression with either OLS or Gradient Descent optimization, the goal is to calculate and find the weight value that give us the minimum loss output value.
2. $argmin_x f(x)$: argmin refers to the value of the argument in a function, that leads to the minimum function output. In this case it is to find the value of x that leads to the lowest f(x) value.
    This is commonly used later on in Coordinate Descent, where we loop through each feature's weight to find the value that minimizes the loss output. 

An example of $min_x f(x)$ and $argmin_x f(x)$ is as below:\
Formula: $f(x)=(x-3)^2$\
$min_x f(x) = 0$: This is because the possible minimum value of f(x) is 0, where $(3-3)^2=0$\
$argmin_x f(x) = 3$: This is because in argmin, we're finding the value of the function argument, in this case x, to reach the minimum value of f(x), where $(3-3)^2=0$

**i. Coordinate Descent**
- In essence, the goal of coordinate descent is to perform soft-thresholding on loop across all the features weight, j from 1 to m.
- The step of coordinate descent is as below:
1. For features weight, $\theta_j$ from j = 1...,m (Tips: We do not start from 0 in weights index, j as j=0 is often reserved for bias in modern notations, i.e. $\theta_0$)
2. We calculate $\frac{1}{n}x^T_j(y-\sum_{k\ne j}^{m}x_k\theta_k)$
3. We perform soft-thresholding on each feature weight, $\theta_j$
4. Step 1-3 forms a complete single iteration in coordinate descent, and we loop though the iteration for k numbers of time, where k = 1,2,3...
- Most importantly, in coordinate descent, we will treat each feature weight individually by separating their loss from the other weights. Thus, the formula, $\sum_{k\ne j}^{m}x_k\theta_k$ which stands for the sum of all weights that are not the jth weight that we're focusing on.
- For example, let's say we have m number of weights. We're focusing on the jth weight, and by doing so we separate it from the all other weight, thus, $k\ne j$, where k is the sum of all weights from 1 to m except j.
- Now if you not know by now, our example of Lasso above is actually a simple implementation of coordinate descent, where we loop through each feature's weight, $\theta_1$ and $\theta_2$ and separate them from each other and treat each of them individually.
- Below is more of a rigorous math intuition towards coordinate descent:

**Coordinate Descent explanation and generalization in Math:**
- The goal of coordinate descent is to minimize the loss function of the Lasso Regression, where we set it as f(x).
```math
min_\theta f(\theta_j)
```
- In Lasso, our loss function is separate into 2 parts, with the first part as the MSE=$\frac{1}{n}(y-X\theta)^2$, we call it $g(\theta)$ here, and Lasso Penalty=$\sum_{j=1}^{m}\lambda|\theta_j|$, we call it $\sum_{j=1}^{m}h(\theta_j)$ here.
- When combined, the final Lasso loss is as below:
```math
f(\theta_j)=g(\theta_j)+h(\theta_j)
```
- Thus, the generalized formula for coordinate descent is:
```math
\theta_j^{(k)}=argmin_{\theta_j} f(\theta_1^{(k)}, \theta_2^{(k)},...,\theta_{j-1}^{(k)}, \theta_j, \theta_{j+1}^{(k-1)},...\theta_m^{(k-1)})
```
**Where:**
$\theta_j$: Weight for jth feature
$f(\theta_1, \theta_2,...,\theta_{j-1}, \theta_j, \theta_{j+1}, \theta_{m})$: Loss function of Lasso with m total weights as parameters.
j: Index of features where j=1,2,3...,m
k: Iterations of the coordinate descent. This is more complicated and will cover below as we're using "one-at-a-time" update than "all-at-once" update.

**Explanation of iterations in coordinate descent:**
- If you look at the generalized formula above, you'll notice that the weights before index j have the iteration k, while the weights after j have the iteration k-1.
- This is because while we're calculating each weights from index j=1,...,m, some of the weights before the jth weight has been updated with the latest value while the weights after the jth weight has not been updated, thus we're using the old index.
- To illustrate here's the scenario below:
- Scenario (3 weights, $\theta_1$, $\theta_2$, $\theta_3$ 5 iterations, k=1,...,5)
  - While we're at the last iteration, k = 5, we begin updating the first weight, $\theta_1$
  - While calculating for newest $\theta_1$ value, $\theta_1^{(5)}$, we need the weight value of $\theta_2$ and $\theta_3$ as well.
  - However, since we're just beginning the 5th iteration and the weights after $\theta_1$ has not been updated yet, thus there's no $\theta_2^{(5)}$ and $\theta_3^{(5)}$
  - Thus, we will use the value of the other weights from the previous iterations, k = 4. We get $\theta_2^{(4)}$ and $\theta_3^{(4)}$
  - As a result, when j = 1, we get the notation: $f(\theta_j, \theta_{j+1}^{(k-1)}, \theta_{j+2}^{(k-1)})$
  - This is why you see that the above generalized formula, $f(\theta_1^{(k)}, \theta_2^{(k)}, \theta_{j-1}^{(k)}, \theta_j, \theta_{j+1}^{(k-1)})$, when we're updating the jth weight, the weights before index j (1, 2,..., j-1) that has been updated will use the latest iteration value, $\theta_1^{(k)}$, $\theta_2^{(k)}$,...,$\theta_{j-1}^{(k)}$, while the weights after index j that has not been updated (j+1,...,m) will use the previous iteration value, $\theta_{j+1}^{(k-1)}$, $\theta_m^{(k-1)}$
  - Lastly, we do not put any index on the weight we're updating, i.e. $\theta_j$ as we're updating its value for the latest iteration. Thus, there's no need for it.

- Below is the steps of updating each weight of feature, $\theta_j$ using coordinate descent:
**Steps in Coordinate Descent:**

**1. Modified gradient MSE loss for coordinate descent**
- In this part, we will be modifying the gradient loss by adjusting from getting the entire sum of the loss of all weights to separating each individual weights loss from the entire weights loss in OLS, which is shown as below:

```math
\begin{aligned}
& \frac{\partial J(\theta)^{MSE}}{\partial \theta_j}=-\frac{1}{n}x^T_j(y-x_j\theta_j)\\
&= -\frac{1}{n}x^T_j(y-(x_j\theta_j+\sum_{k\ne j}^{m}x_k\theta_k))\\
&= -\frac{1}{n}x^T_j(y-(x_j\theta_j+x_{k\ne j}\theta_{k\ne j}))
\end{aligned}
```

**2. Combining modified MSE gradient with Lasso:**
```math
\begin{aligned}
& \frac{\partial J(\theta)^{Lasso}}{\partial \theta_j}=-\frac{1}{n}x^T_j(y-(x_j\theta_j+x_{k\ne j}\theta_{k\ne j}))+\lambda\cdot \frac{\partial }{\partial \theta}(|\theta|_1)\\
&= -\frac{1}{n}x^T_jy+\frac{1}{n}x^T_jx_j\theta_j+\frac{1}{n}x^T_jx_{k\ne j}\theta_{k\ne j}+\lambda\cdot \frac{\partial }{\partial \theta}(|\theta|_1)\\
& \text{Let } p_j= \frac{1}{n}x^T_jy-\frac{1}{n}x^T_jx_{k\ne j}\theta_{k\ne j},\\
&= -p_j+\frac{1}{n}x^T_jx_j\theta_j+\lambda\cdot \frac{\partial }{\partial \theta}(|\theta|_1)
\end{aligned}
```

**3. Set gradient to be 0:**
```math
\begin{aligned}
& \frac{\partial J(\theta)^{Lasso}}{\partial \theta_j}=-p_j+\frac{1}{n}x^T_jx_j\theta_j+\lambda\cdot \frac{\partial }{\partial \theta}(|\theta|_1)\\
& 0 = -p_j+\frac{1}{n}x^T_jx_j\theta_j+\lambda\cdot \frac{\partial }{\partial \theta}(|\theta|_1)\\
&= \begin{cases} -p_j+\frac{1}{n}x^T_jx_j\theta_j-\lambda& \text{if } \theta_{j} < 0 \\ {[-p_j-\lambda, -p_j+\lambda]} & \text{if } \theta_{j} = 0 \\ -p_j+\frac{1}{n}x^T_jx_j\theta_j+\lambda& \text{if } \theta_{j}> 0 \end{cases}\\
\end{aligned}
```

**4. Soft Thresholding**
Thus, when we're doing soft thresholding, we can solve for $\theta_j$ for each case:

**When $\theta_j<0$,**
```math
\begin{aligned}
& -p_j+\frac{1}{n}x^T_jx_j\theta_j-\lambda = 0\\
& \frac{1}{n}x^T_jx_j\theta_j = p_j+\lambda\\
& x^T_jx_j\theta_j = np_j+n\lambda\\
& \theta_j = (np_j+n\lambda)(x^T_jx_j)^{-1}, p_j < -\lambda
\end{aligned}
```
**Additional Note:** The reason we set the condition $p_j < -\lambda$ is to ensure that the final expression value is in negative, which ensures that $\theta_j$ < 0 and not flipping it to be $\theta_j > 0$, which violates the KKT condition.

**When $\theta_j >0$,**
```math
\begin{aligned}
& -p_j+\frac{1}{n}x^T_jx_j\theta_j+\lambda = 0\\
& \frac{1}{n}x^T_jx_j\theta_j = p_j-\lambda\\
& x^T_jx_j\theta_j = np_j-n\lambda\\
& \theta_j = (np_j-n\lambda)(x^T_jx_j)^{-1}, p_j > \lambda
\end{aligned}
```
**Additional Note:** Similarly above, the reason we set the condition $p_j > \lambda$ is to ensure that the final expression value is in positive, which ensures that $\theta_j$ > 0 and not flipping it to be $\theta_j < 0$, which violates the KKT condition.

**When $\theta_j$=0,**
```math
\begin{aligned}
& 0 \in [-p_j-\lambda, -p_j+\lambda]\\
& -p_j-\lambda \leqslant 0 \leqslant -p_j+\lambda\\
& -\lambda \leqslant p_j \leqslant \lambda
\end{aligned}
```

Thus, we can rewrite this as:
```math
\begin{aligned}
\begin{cases} \theta_j=(np_j+n\lambda)(x^T_jx_j)^{-1}& \text{if } p_j < -\lambda \\ \theta_j=0 & \text{if } -\lambda \leqslant p_j \leqslant \lambda \\ \theta_j=(np_j-n\lambda)(x^T_jx_j)^{-1}& \text{if } p_j > \lambda \end{cases}\\
\end{aligned}
```

This can be summarized into a soft thresholding summation:
```math
\theta_j=n(x^T_jx_j)^{-1}\cdot S(p_j, \lambda)
```
Where $S(p_j, \lambda)$ is the soft thresholding function notation, scaled to n the total number of dataset and the inverse of OLS matrix $(x^T_jx_j)^{-1}$. To be more specific, the $S(p_j, \lambda)$ can be written as:
```math
S(p_j, \lambda)=sign(p_j)\cdot max(|p_j|-\lambda, 0)
```
This matches with the earlier example we use, $max(2-\lambda, 0)$.

**Conclusion of Coordinate Descent**
In short, in coordinate descent, it allows Lasso to calculate the partial error of the coefficient (focused individual weights) and reduce its value by $\lambda$, which is the regularization constant. Based on the formula, if the focused individual weight, $\theta_j$ is extremely small, it causes the residual of the specific feature, $\frac{1}{n}x^T_jx_j\theta_j$ to be smaller than $\lambda$. When $\frac{1}{n}x^T_jx_j\theta_j < \lambda$, the result will be in negative value. This causes the max() function to activate which turn the feature weight to be exactly 0. This induces sparsity and perform feature selection by dropping out the irrelevant feature with extremely small weights.

**Additional Notes:**
- In some academic papers or lecture notes, you might see that the soft thresholding function is written as:
```math
\theta_j=S(p_j, \lambda)
```
- This can be explained in 2 reasons:
1. In their OLS gradient, they do not scale by n. Thus, there's no $\frac{1}{n}$ in their OLS gradient loss and there'll be no n in the final expression. The reason why we choose to scale by n is to prevent the gradient from exploding as we're calculating the dot product across all datasets. Thus, if we do not scale by n to get the average of the gradient, the gradient of 1,000,000 datasets will be astronomically larger than 1,000 datasets. 
2. On the other hand, for simplification reasons, many academic examples use orthogonal matrix. As explained above in our Lasso example where we use orthogonal matrix X as well, the beauty lies in that the dot product of an orthogonal matrix with its transpose is exactly the identity matrix, I. Thus, the inverse of $X^TX$, $(X^TX)^{-1}$ will be just I, and scale it by n it'll be nI. Thus, the final expression will be as below:
```math
\begin{aligned}
& \begin{cases} \theta_j=(np_j+n\lambda)& \text{if } p_j < -\lambda \\ \theta_j=0 & \text{if } -\lambda \leqslant p_j \leqslant \lambda \\ \theta_j=(np_j-n\lambda)& \text{if } p_j > \lambda \end{cases}\\ 
& \\
& \begin{cases} \theta_j=n(p_j+\lambda)& \text{if } p_j < -\lambda \\ \theta_j=0 & \text{if } -\lambda \leqslant p_j \leqslant \lambda \\ \theta_j=n(p_j-\lambda)& \text{if } p_j > \lambda \end{cases}\\ 
& \\
& \theta_j=n\cdot S(p_j, \lambda)
\end{aligned}
```
And since the examples does not scale by n, thus without n at the end it'll just be:
```math
\theta_j=S(p_j, \lambda)
```
3. Lastly, in our Lasso example above, we further simplify the example by setting the other weights as 0, $\sum_{k\ne j}^{m}x_k\theta_k=0$ while focusing on the individual weight, $\theta_j$. For example, when we focus on the first feature weight, $\theta_1$, the other weight, $\theta_2$ is 0 and vice versa. In practical coordinate descent we do not set other weights as 0, as we simply just let the other weights remain their as-if value.

**Stochastic Gradient Descent vs Coordinate Descent**:
The main difference between Coordinate Descent and Gradient Descent lies within their steps in optimizing the weight value, $\theta$. Here are the comparison of steps between them:

**1. Stochastic Gradient Descent:**
- a) We first separate the entire dataset into small batches, known as mini-batch, i
- b) For each mini-batch, i, where i=0,1...,n, we calculate the gradient of the MSE loss, $\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})x_{ij}$ in summation form or $\frac{1}{n}x_j^T(\hat{y}-y)$ in matrix form
- c) We then perform gradient descent update by reducing the weight values from the gradient loss multiplied with learning rate, $\alpha$, $\theta = \theta-\alpha\frac{\partial J(\theta)}{\partial \theta}$
- d) We then loop through the entire mini-batches to complete one iteration, or known as 1 epoch

**2. Coordinate Descent:**
- a) For features weight, $\theta_j$ from j = 1,...,m 
- b) We modify the gradient of MSE by separating each individual feature in loop from the other sum of features, $\frac{1}{n}x^T_j(y-\sum_{k\ne j}^{m}x_k\theta_k)$
- c) We calculate the soft-thresholding through sub-gradient and KKT conditions on each feature weight, $\theta_j$
- d) We update each feature weight based on the soft-thresholding to complete 1 iteration, or known as 1 epoch 

Lastly, the reason we prefer `Coordinate Descent` over `Stochastic Gradient Descent (SGD)` is because: 

In SGD, we reduce the weight value approximately, which causes the weight to oscillate around 0 and not completely 0 (i.e. 0.0000001 or -0.0000001). Thus, this does not induce sparsity as expected from Lasso which turn the irrelevant feature's weight into exactly 0.

On the other hand, in Coordinate Descent, thanks to the soft thresholding function, which introduces max($p_j-\lambda$, 0). Based on the formula, if the weights value is too small, it proves that the feature is irrelevant and Lasso will perform feature selection by turning the weight to be exactly 0, dropping the feature column in the process. As a result, this achieves the proof-of-concept of sparsity in Lasso.

**ii. Proximal Gradient Descent**
- In proximal gradient descent, it is used to approximately minimize a function where it consists of both smooth and non-smooth parts, where one part of the function is differentiable, while the other part is non-differentiable.
- This means that when we're splitting a function, f($\theta$) into 2 parts, namely g($\theta$) and h($\theta$) the first part of the function, g(x) is differentiable while the other part, h(x) is non-differentiable.
- Let's take an example of the Lasso regression loss function:
```math
f(\theta)=\left( \underbrace{\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}}_{\text{g(}\theta\text{) - Differentiable}}+ \underbrace{\lambda\sum_{i=1}^{m}|\theta_{i}|}_{\text{h(}\theta\text{) - (Non-Differentiable)}} \right)
```
- When we rewrite it in matrix form:
```math
f(\theta)=\left( \underbrace{\frac{1}{2n}||y-X\theta||^2_2}_{\text{g(}\theta\text{)}}+ \underbrace{\lambda||\theta||_1}_{\text{h(}\theta\text{)}} \right)
```

Where:
$f(\theta)$: The Lasso Loss function\
$g(\theta)$: Mean Square Error, MSE\
$h(\theta)$: Lasso Regularization Penalty\
$\lambda$: Regularization constant\
$||\theta-\theta_{new}||^2_2$: Calculate the distance between the old weight and the new weight, which is equivalent to: $(\sqrt{\theta^2-\theta_{new}^2})$. The subscript 2 represents the L2 Normalization, which is known as Euclidean Norm\
$||\theta||_1$: The subscript 1 represents the L1 norm of the weights, which is equivalent to $\sum_{j=1}^{m}|\theta_j|$

**Additional Notes:** 
1. For many cases here you'll see that we're switching to matrix notation, as the author believe it'll be easier to understand and transition well to practical code since we'll be using numpy for vector and matrix manipulation of X, y, weights and bias.
2. `Proximal Gradient Descent` is a more general approach used to deal with functions that has differentiable and non-differentiable parts. For Lasso Regression, we'll be using a more specific technique, which is known as `Iterative Soft Thresholding Algorithm (ISTA)`, a subset of Proximal Gradient Descent which concepts is shown as below.

As you can see, the g($\theta$) part of the Lasso function which is the MSE is differentiable, while h(x) part which is the Lasso penalty is non-differentiable as it is a convex function with subgradient at 0.

Thus, the steps of `Proximal Gradient Descent`, specifically `Iterative Soft Thresholding Algorithm (ISTA)` at minimizing functions similar to this is as below:
1. Break the function, f($\theta$) into differential and non-differential parts, g($\theta$) and h($\theta$)
2. Perform basic gradient descent on g($\theta$) to update the weights which helps to minimize the Lasso loss, f($\theta$).
3. By using the previous weights value, we compute the proximal operator to calculate the new weights that minimizes the h(x) value while ensuring the value is close to the previous weights.
4. Iterate the steps from 2 to 3 until the function converges, i.e. it hits the minimum loss value where f($\theta$) $\approx$ 0.

**Additional Notes:** Since h($\theta$) is non-differentiable at 0, we need to use a different approach than gradient descent to ensure it really hits 0 instead of oscillating around it. Thus, in proximal gradient descent, we will implement soft thresholding which update the weights value to exactly 0 under certain conditions

Here's the mathematical steps for the ISTA:

a) Break the function into differential and non-differential parts
```math
f(\theta)=\left( \underbrace{\frac{1}{2n}||y-X\theta||^2_2}_{\text{g(}\theta\text{)}}+ \underbrace{\lambda||\theta||_1}_{\text{h(}\theta\text{)}} \right)
```

b) Update the weights, $||\theta||$ using basic gradient descent on g($\theta$)
```math
\begin{aligned}
& \theta = \theta - \alpha\frac{\partial }{\partial \theta} \frac{1}{2n}(y-X\theta)^2\\
& = \theta + \alpha\frac{1}{n}X^T(y-X\theta)
\end{aligned}
```

c) Compute the new weights using previous vector values and proximal operator to minimize f($\theta$) value
The general proximal operator for non-differential part, h(x) in Proximal Gradient Descent is as below:
```math
prox_{h,\alpha}(x) = argmin_z\frac{1}{2\alpha}||z-x||^2_2+h(z)
```
Where:\
h: h(z) non-differential part\
$\alpha$: Learning rate\
x: Weight vector\
z: New weight vector, you can think of it as $x^{new}$\
$argmin_z$: Finding the value of vector z that minimizes the h(x) value.

For better understanding, you can imagine vector $\theta$ as a group of points in the cartesian plane, and your goal is to find the new group of points, vector z, which helps reduce the h($\theta$) value such that it is minimized. For example:
```math
z-x = x^{new} - x^{old} = \text{min } h(x)
```

Now we can plug it in into our ISTA with f($\theta$) Lasso loss function:
```math
\begin{aligned}
& \text{Consider:}\\
& f(\theta) = \left( \underbrace{\frac{1}{2}||y-X\theta||^2_2}_{\text{g(}\theta\text{)}}+ \underbrace{\lambda||\theta||_1}_{\text{h(}\theta\text{)}} \right)\\
& \text{Then: }\\
& prox_\alpha(\theta) = argmin_{\theta_{new}} \frac{1}{2\alpha}||\theta-\theta_{new}||^2_2+\lambda||\theta_{new}||_1\\
&= argmin_{\theta_{new}} \frac{1}{2}||\theta-\theta_{new}||^2_2+\alpha\lambda||\theta_{new}||_1\\
&= S_{\lambda}(\theta)
\end{aligned}
```
Where:
$prox_\alpha(\theta)$: The proximal operator with respect to the weights, $\theta$ using the learning rate, $\alpha$

After reading Coordinate Descent, the final minimizing formula might see familiar to you:
- Coordinate descent:
```math
\theta_j^{(k)}=argmin_{\theta_j} f(\theta_1^{(k)}, \theta_2^{(k)},...,\theta_{j-1}^{(k)}, \theta_j, \theta_{j+1}^{(k-1)},...\theta_m^{(k-1)})
```
where f($\theta_j$) = g($\theta_j$) + h($\theta_j$)
- Proximal Gradient Descent (ISTA):
```math
\theta = argmin_{\theta_{new}} \frac{1}{2}||\theta-\theta_{new}||^2_2+\alpha\lambda||\theta_{new}||_1
```
where f($\theta$) = g($\theta$) + h($\theta$)

Where the goal for both optimization algorithms is to calculate the weight values that minimizes the Lasso loss function, as seen in $argmin_{\theta_{new}}$

Thus, similar to coordinate descent, we will also derive a soft thresholding function for proximal gradient descent as well:
```math
\theta = S_{\lambda\alpha}(\theta)
```
Which can be derived into:
```math
\begin{cases} S_{\lambda\alpha}(\theta)_j=(\theta_j-\alpha\lambda)& \text{if } \theta_j > \alpha\lambda \\ \theta_j=0 & \text{if } -\alpha\lambda \leqslant \theta_j \leqslant \alpha\lambda \\ \theta_j=(\theta_j+\alpha\lambda)& \text{if } \theta_j < -\alpha\lambda \end{cases}
```

Lastly, recall that we have calculate the gradient descent of g($\theta$) to update the weights, we will be adding them into the proximal operator, specifically in the soft thresholding as well. This is to ensure that the updated weights is close to the minimum of g($\theta$) loss.
```math
\begin{aligned}
& \theta_{k+1} = S_{\lambda\alpha}(\underbrace{\theta_k}_{\text{Proximal Operator}} - \underbrace{\alpha\nabla g(\theta_k)}_{\text{Gradient Descent of g(}\theta\text{)}})\\
&= S_{\lambda\alpha}(\theta_k + \frac{\alpha}{n} X^T(y-X\theta_k)
\end{aligned}
```

**Additional Note:**
1. The reason that we're dividing with the learning rate, $\alpha$ in the proximal operator is based on the `decomposition of quadratic approximation` using Taylor Series expansion for gradient descent. From there you'll understand how proximal operator of h(x) can still ensure the point we're finding is close to g(x) gradient descent, which is as follow:
```math
\theta = argmin_{\theta_{new}} \frac{1}{2\alpha}||\theta_{new}-(x-\alpha\nabla g(\theta))||^2_2 + h(\theta_{new})
```
This generalized formula is where our Lasso version of proximal operator is derived from. The first term, $\frac{1}{2\alpha}||\theta_{new}-(x-\alpha\nabla g(\theta))||^2_2$ causes the distance between the new weight vector, z and the old weight vector, $\theta$ to be minimum through $argmin_z$ function and add it with h($\theta$) the non-differential part
2. In some papers you might see the coordinates of points being flipped, such as $||\theta_{new}-\theta||^2_2$ or $||\theta-\theta_{new}||^2_2$. Both of them are the same as recall in calculating the distance between points, we calculate the magnitude which ignores positive or negative values

d) Repeat step 2-3 until convergence
- In the last step, similar to coordinate descent and stochastic gradient descent, step 2-3 represent a complete iteration of the optimization algorithm and we will be repeating this for n iterations of time, or known as epoch until the Lasso model convergence (i.e. loss curve becomes stable)

**Difference between Coordinate Descent and Proximal Gradient Descent (ISTA)**
1. Coordinate Descent
- Update each feature's weight one at a time inside an iteration
- Normally used to achieve sparsity, which in this case perfect for L1 Lasso
- Complexity is O(mn), as it loops through each feature individually for a complete cycle. However, it might be easier to compute as it "updates" one feature at a time, making it more efficient as it calculates partial of a full vector weight gradient 
2. Proximal Gradient Descent (ISTA)
- Update the entire features' weights as a vector at once within a single iteration
- Works well with composite functions, in this case it suits L1 Lasso as well due to the combination of differentiable and non-differentiable parts, h($\theta$) and g($\theta$)
- While the complexity is the same with O(mn), it is more memory intensive at times due to it calculating the entire full vector weight gradient, causing it to struggle at larger datasets with many features.

In short, coordinate descent and Proximal Gradient Descent with ISTA is similar in essence and their backbone mechanisms highly overlap, the major difference is their approach to update the weights and the complexity, where the difference is slight and can vary based on the dataset scenarios. 

**Conclusion of ISTA and Proximal Gradient Descent**
In short, we have applied Proximal Gradient Descent, specifically ISTA into Lasso, where it updates all features' weights in a dataset at once and apply soft thresholding in the weights. For weights whose value is smaller than the regularization constant, $\lambda$, it will be changed directly to 0.

**iii. Least Angle Regression (LARS)**
- Onto the last optimization algorithm, the Least Angle Regression (LARS) is a method that is used to trace all of the possible solution paths for a regression models.
**Disclaimer:** The math behind LARS is too complex as it involves rigorous geometrical angle analysis and more deep math, so for the sanity this will not be covered. However, below is the general intuition of LARS:

**In-depth explanation of LARS:**
1. In LARS, it will first calculate the correlation between each weight and the target output (i.e. house size(feature) vs house price(target))
2. Then, after LARS has found the optimum feature with high correlation with the target, it will "step" more into the target and adding more other features into it, creating a path
3. However, unlike previous optimizations like `Coordinate Descent` and `Proximal Gradient Descent & ISTA`, it does not fully commit into that best possible path. Instead, it will go through other possibility of paths by combining different other features together, just that it will pay less attention to it.
4. As LARS continues, it will add more features into the path and determine if the correlation of the newly added features is similar with the correlation of the current highest one. If it is, it will move to the direction of the angle in between the 2 variables, which is known as finding the equiangular vector.
5. Ths logic behind is that if the features that are highly correlated is located at this direction, LARS will move towards that direction, specifically in between the angles as naturally other highly correlated features will be located there as well.
6. As the number of features increase, LARS will expand the paths by extending more features into it, creating more possibilities and eventually result in a web of paths. It will continue the pattern of step 4-5. 
7. At the same time, the paths with variables that are less correlated, meaning they do not contribute much to the overall target(price), they will be slowly being less attracted by LARS, causing their weights value to become smaller and eventually shrink to 0, which induces sparsity as expected in Lasso.
8. Lastly, when all of the features are included by LARS, it will choose the most optimum path that contains all of the features with the highest correlation with the target.

**Conclusion of LARS**
In short, LARS is a more "democratic" algorithm compared to previous optimization algorithms, where it considers all of the possibility between the features combination instead of fully committing into one path only. This allows LARS to uncover the full paths of the Lasso Regression. This allows LARS to provide a full Lasso solution unlike the others.

**Advantage of LARS**
1. More computational efficient: LARS is considered more efficient in datasets that has more features as it is good at expanding through numerous features and their correlation with the output
2. Full Lasso solution: LARS is able to provide a full view of possible solutions for Lasso Regression due to its ability to sought after all possibilities of features paths.

**Disadvantage of LARS**
1. Noise Sensitivity: LARS is more sensitive to the noise, as it does not check for correlation between the features themselves, causing it to struggle in multicollinearity problem
2. Complexity: LARS is a pain to implement due to its complex algorithm structure and rigorous math.

**Differences between LARS and Coordinate Descent & Proximate Gradient Descent**
1. LARS
- It is more computational efficient in certain dataset scenarios (i.e. high number of features dataset). However, its computational power might be higher when with high data samples, n and low features, m.
- It covers other possibilities of features combination paths instead of fully commit into the best path, making it able to provide full solution view due to its "democratic" nature where it adds the features gradually one by one, instead of rushing into conclusion with just 1 feature information
2. Coordinate Descent & Proximate Gradient Descent
- In some scenarios it might consume more computational power as they either update all features weights at once or checking all of the weights, causing them to struggle with high characteristics dataset. However, it is generally faster as a more general optimization with high data samples, n and low data features, m
- It fully commits into the most optimized solution by analyzing each individual feature only and ignore the other features contribution at a time, making it a more "greedy" forward selection than LARS

**When to use which optimization algorithms:**
1. Use LARS when:
- You require a full and complete feature weights solution path with the cheapest computational method.
- You require feature explanability and intepretation to explain why we select the specific feature and drop the others through sparsity
- The model is not too complex, i.e. the number of features is low with high data samples

2. Use Coordinate Descent when:
- The model is complex with high dimensional dataset: Coordinate Descent is very effecient in calculating the single optimized solution
- You're not too interested in finding the full solutions and only keen on the single optimized solution
- You're using Cross Validation: Coordinate Descent performs well when combining with cross validation to choose the hyperparameter, as shown in Scikit-Learn where Coordinate descent is chosen as the general optimization algorithm

3. Use Proximate Gradient Descent when:
- You require parallelism between software and hardware
- However, Proximal Gradient Descent is less commonly used as it is only suitable for simple dataset with smaller features, and it is harder to implement than coordinate descent

In conclusion, use LARS when you're facing a very niche problem and require very high feature selection explanability. Else pick Coordinate Descent as it is commonly used and easier to implement than the rest.

# Reference
1. Tutorial on Lasso (Statistics Student Seminar @ MSU): https://www.stt.msu.edu/users/wangho16/lasso.pdf
2. CMU Proximal Gradient Descent Notes (Lecture 8:  Convex Optimization 10-725): https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/08-prox-grad-scribed.pdf
3. CMU Coordinate Descent Notes (Convex Optimization 10-725): https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/coord-desc.pdf
4. Proximal Gradient Descent & Acceleration (Convex Optimization 10-725): https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf
5. Coordinate Descent for Lasso: https://stats.stackexchange.com/questions/347796/coordinate-descent-for-lasso
6. Understanding LARS Lasso: https://www.geeksforgeeks.org/machine-learning/understanding-lars-lasso-regression/
7. Getting to know LARS (Least Angle Regression): https://medium.com/@hannah.hj.do/getting-to-know-lars-least-angle-regression-f50e94c34b97
