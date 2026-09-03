# Chapter 4.3: Lagrangian Duality and Karush-Kuhn-Tucker (KKT) Conditions
- In this chapter, we'll be covering:
1. Lagrangian Duality and how it is related to Lasso optimization problem
2. Generalization of KKT Conditions from Lagrangian Duality

# Lagrangian Duality
For deep learners and people that appreciate statistical analysis who want to know how we defined the optimized value for the subgradient of Lasso to be 0, we need to go all the way back into Lagrangian Duality. Consider a set of problem as below:
```math
\begin{aligned}
& min_\theta f(\theta)\\
& \text{Such that: }\\
& g(\theta) \le 0\\
\end{aligned}
```
Where:
f($`\theta`$): Lasso MSE part\
$`g(\theta)`$: Lasso Penalty part\

As you see, this problem set is just as similar with our Lasso Regression algorithm as below:
```math
min_{\theta} \frac{1}{2n}(||y-\hat{y}||^{2}) + \lambda||\theta||_1
```
Where we'll visualize this further as:
```math
\begin{aligned}
& \underbrace{min_{\theta}\frac{1}{2n}(||y-\hat{y}||^{2})}_{\text{f(}\theta\text{) - MSE Part in Lasso Loss}} \\
& \text{Such that: }\\
& \underbrace{(||\theta||_1 - t) \le 0}_{\text{g(}\theta\text{) - Absolute Value Function Lasso Penalty}}\\
\end{aligned}
```
Notice that in our problem, we add a t into our $`g(\theta)`$ function where $`|\theta|`$ becomes $`(|\theta| - t)`$. This is because we're introducing constrained function by subtracting weights with a constraint t, instead of non-constrained function like $`\lambda||\theta||_1`$ into our Lagrangian Multiplier

In this scenario, t act as a budget constraint which act as a cap limit for the total sum of the weights, $`\sum_{j=1}^{m}|\theta_j|`$

Before we move on to define our Lagrangian, we need to know the following terms and definition:
1. Convex function: Function that represents a bowl-like shape, such as a positive hyperbola. I.e.: $`f(x) = x^2`$, $`f(x) = |x|`$, etc
2. Concave function: Function that represents an inverted bow-like shape, such as a negative hyperbola. I.e.: $`f(x) = -x^2`$, $`f(x) = -|x|`$, $`f(x) = ln(x)`$, etc
3. Affine function: Any linear function that resembles the straight line formula, $`y=mx+c`$

Thus,
$`f(\theta)`$: Convex function (The square quadratic function resembles a bowl-like shape)\
$`g(\theta)`$: Convex function (The absolute value function resembles a bowl-like shape)\

# Primal Lagrangian Function
We then can define this problem set into a standard Primal Lagrangian as:
```math
\mathcal{L}(\theta, \lambda) = f(\theta) + \lambda g(\theta)
```
Where:
$`\mathcal{L}`$: Lagrangian function\
$`f(\theta)`$: The MSE part in Lasso loss function, which is a convex function\
$`\theta`$: Weights vector with shape of (1, m)\
$`\lambda`$: Lagrangian multiplier

When we plug this into our Lasso Problem:
```math
\begin{aligned}
& \mathcal{L}(\theta, \lambda) = f(\theta) + \lambda g(\theta)\\
&= \underbrace{\frac{1}{2n}(||y-\hat{y}||^{2})}_{\text{f(}\theta\text{)}} + \underbrace{\lambda(||\theta||_1 - t)}_{\lambda g(\theta)}\\
&= \underbrace{\frac{1}{2n}(||y-\hat{y}||^{2}) + \lambda||\theta||_1}_{\text{Standard Lasso Loss}} - \underbrace{\lambda t}_{\text{Constant}}
\end{aligned}
```
The standard Primal Lagrangian looks similar to our Lasso Loss function with an extra constant, $`\lambda t`$

Where this is known as a standard Primal Lagrangian Function.

For those that were not familiar with what is Lagrangian and Lagrangian multiplier, below is a simple explanation:
- Lagrangian is a general concept/method for optimization problem, which can be found in Physics Energy and Mechanisms chapter to find the least amount of energy consumption
- Lagrangian multiplier is a relative field of Lagrangian, where it is a mathematical model used to calculate the minimum or maximum of a function. In this case we'll be using Lagrangian Multiplier as it helps us to model to find the local minima or maxima of our Lasso Regression loss problem.

**Additional Note:** You might now realise that our Lagrangian Multiplier, $`\lambda`$ is similar as our Lasso Regularization Hyperparameter, $`\lambda`$, and that is true. This is because in convex functions like L1 Lasso and L2 Ridge that came from the Lagrange Duality, the Regularization Hyperparameter is closely related to the Lagrange Multiplier which is controlled based on the budget constraint, t.

Importantly, you need to know that we must follow the constraints of our Primal Lagrangian Problem, such that $`g{\theta} \le 0`$. If not, it will violate the primal Lagrangian function and our value will be positive infinity such that:
```math
\begin{aligned}
& \text{if } g(\theta) > 0,\\ 
&  max_{\lambda; \lambda \ge 0} f(\theta) + \lambda g(\theta) = ∞
\end{aligned}
```
As a result, our Primal Lagrangian Constraint will be:
```math
\begin{aligned}
& max_{\lambda; \lambda \ge 0} f(\theta) + \lambda g(\theta)\\
& = \begin{cases} f(\theta)& \text{if }\theta\text{ satisfies the primal constraint}\\ ∞& \text{otherwise} \end{cases}\\
\end{aligned}
```
Thus, when we're trying to maximize the Primal Lagrangian function, our $`g(\theta)`$ is at most equals to 0, as per the constraints of our model problem.

You can consider this as:
```math
\begin{aligned} 
& max_{\lambda; \lambda \ge 0} \mathcal{L}(\theta, \lambda)\\
&= max_{\lambda; \lambda \ge 0} f(\theta) + \lambda g(\theta)
\end{aligned}
```
Where we're trying to maximize our Lagrangian Multipliers constant such that our $`g(\theta)`$ will be 0, which allows our Primal Lagrangian will be equal to the Lasso Loss function, f($`\theta`$).

Additionally, there's also Constraint in our Dual Lagrangian Problem as well, where:
```math
\lambda \ge 0
```
This is why you'll see in the max function we put a condition where the Lagrange Multiplier, $`\lambda`$ must be greater or equal to 0

For both Primal and Dual Lagrangian Problem, we must follow their constraint in order for the problem to be solvable.

However, remember that our goal is to minimize our Lasso loss function so that the optimal minimum loss will be found. As a result, the final primal Lagrangian solution should be:
```math
min_{\theta} max_{\lambda; \lambda \ge 0} \mathcal{L}(\theta, \lambda) := p^*\\
```
Where we are minimizing our Lasso loss and at the same time maximize the Lagrangian Multiplier to get the most optimal solution in Primal Lagrangian. We'll create a notation and call it $`p^*`$, and set its value to the most optimal solution in Primal Lagrangian.

# Dual Lagrangian Function
In Dual Lagrangian, we'll be taking our Primal Lagrangian solution and convert it slightly. So we'll take the final solution, and flip the min and max position as below:
```math
g(\lambda) = min_{\theta} \mathcal{L}(\theta, \lambda)\\
```
Geometrically, the reason we flip the min and max position is that in Dual Lagrangian, g($`\lambda`$), it is always a concave function even though the $`f(\theta)`$, $`g(\theta)`$ in primal Lagrangian is convex function.

Since the Dual Lagrangian is concave, we'll be maximizing the solution as it helps to find the optimal solution in Primal Lagrangian. The reason is because the maximum point of a concave function is equal to the minimum point of a convex function. (i.e: minimum point of $`f(x) = x^2`$ is equal to the maximum point of $`f(x) = -x^2`$ which is 0).

Furthermore, depends on the problem scenario, one might be harder to find its optimal solution by minimizing/maximizing it. For example, it might be harder to minimize primal than maximize dual, and vice versa.

As a result, we will be using the notation $`d^*`$ to indicate the dual Lagrangian function:
```math
d^* = max_{\lambda; \lambda \ge 0} g(\lambda)
```

Thus, here's the finalized primal and dual Lagrangian function:
```math
\begin{aligned}
& d^* = max_{\lambda; \lambda \ge 0} min_{\theta} \mathcal{L}(\theta, \lambda)\\
& p^* = min_{\theta} max_{\lambda; \lambda \ge 0} \mathcal{L}(\theta, \lambda)
\end{aligned}
```

Now, to form the Dual Lagrangian problem set, we'll arrange both $`p^*`$ and $`d^*`$ as below:
```math
\begin{aligned}
& max_{\lambda; \lambda \ge 0} min_{\theta} \mathcal{L}(\theta, \lambda) \le min_{\theta} max_{\lambda; \lambda \ge 0} \mathcal{L}(\theta, \lambda)\\
& d^* \le p^*
\end{aligned}
```
**Additional Notes:** The reason we're putting the dual Lagrangian, $`d^*`$ to be lesser or equal to the primal Lagrangian, $`p^*`$ is due to the maximum of a minimum function is always lesser or equal to the minimum of a maximum function

This forms the duality problem in Lagrangian, where we'll separate them based on weak duality and strong duality, and explain as below:
1. Weak Duality: Weak Duality occurs when our Dual Lagrangian, $`d^*`$ is lesser or equal to our Primal Lagrangian, $`p^*`$, such that $`d^* \le p^*`$.
2. Strong Duality: However, in some scenarios where we have $`d^* = p^*`$, then we can hold a Strong Duality problem, thanks to Slater's Condition

**Slater's Condition:** When we have a Primal Lagrangian Problem such that it is convex, there exists an $`x^*`$ that is strictly feasible, then strong duality holds.

- Since in our Primal lagrangian Problem, our Lasso is convex, and our Primal Constraint, where $`||\theta||_1 - t \le 0`$, then $`||\theta||_1 \le t`$ which indicates a strict feasible value, where $`x^*`$ = 0.
- As a result, strong duality is valid and applicable in our Lasso Problem
- This means that we have a zero duality gap where $`d^* \le p^*`$ is the same as $`d^* = p^*`$, where the inequality (less than or equal to) is the same as equal.
- As a result, at optimal primal-dual pair where our weights, $`\theta^*`$ Lagrangian Multiplier, $`\lambda^*`$ are primal and dual, where both optima of primal and dual are found. This results in:

```math
\begin{aligned}
& \mathcal{L}(\theta^*, \lambda^*) = d^* = p^*\\
&= f(\theta) + \lambda g(\theta)\\
\end{aligned}
```

**Conclusion of Lagrange Duality**:
- The full reasons why we need Lagrangian Multiplier is that it is a tool for us to formulate mathematical models to find the most optimal solution for our problem set. In this case since we're trying to optimize our Lasso Problem, we'll need to use Lagrangian Multiplier to find the local minima and maxima for our problem set.
- As a result, in Primal Lagrangian Function, the term "Primal" represents the original problem set. Thus, when we mean by formulating a standard Primal Lagrangian Function or creating a Primal Problem, we mean by taking the original constrained problem, in this case Lasso and transform it to form a minimum solution using the restriction (Budget Constrain t) 
- Then, from the Primal Solution we'll derive it to form the Dual Lagrangian Function, where we'll flip the minimum of the maximum to transform into maximizing the minimum of the Lagrangian Function. This is because as our original Primal Problem is a convex problem, our Dual Lagrangian will always be concave despite other conditions. Hence the reason behind flipping the minimum and maximum part
- Furthermore, when we have the primal and dual problem, we'll have to make sure if strong duality exists based on the Slater's condition. If strong duality exists, then we can find the optimum solution of both dual and primal by optimizing one another, as $`d^*=p^*`$ in strong duality instead of inequality.
- For example, in strong duality, we can find the minimum of primal problem, which is the optimum of primal by maximizing our dual problem, and find the maximum of dual problem, which is the optimum of dual by minimizing our primal problem.
- Moreover, in different scenarios and problem optimizing our primal might be harder than optimizing our dual, and vice versa. Thus, it'll be easier if we optimize one problem to find the optimum of the other.
Steps of Lagrange Duality:
1. Formulate a problem model using Lagrangian Function to form Primal Lagrangian Function
2. Minimize the Primal Lagrangian to form the optimum solution
3. Transform the Primal Lagrangian to form the Dual Lagrangian, which is by flipping the max and min sign (min max become max min)
4. Check if Strong Duality holds for the Dual Lagrangian Problem. If it exists, we can determine that our weights and lagrangian multiplier are primal and dual
5. Thus, we can optimize both primal and dual by optimizing one another thanks to zero duality gap where primal and dual are equal instead of inequality (Optimize dual = optimize primal, maximize dual = minimize primal)

Lastly, later on you'll see that we'll use a derived form of Lagrange Duality, which is KKT conditions to determine if 0 will be the optimum value for our Lasso Regression problem.

# Karush-Kuhn-Tucker (KKT) Conditions
- In order to determine the value of $`{\partial }{|\theta|}=0`$ at non-differentiable $`\theta`$=0, we need to follow the Karush-Kuhn-Tucker (KKT) conditions, which is derived from the Lagrangian Duality Problem and is used for optimization in convex functions like the Lasso Regularization.
- The general conditions for KKT is as below:
1. Stationarity: $`0 \in \partial f(\theta) + \lambda \partial g(\theta)`$
2. Complementary Slackness: $`\lambda g(\theta) = 0`$, where $`\lambda(||\theta||_1 - t) = 0`$
3. Primal Feasibility: $`g(\theta) \le 0`$, where $`||\theta||_1 - t \le 0`$, $`||\theta||_1 \le t`$
4. Dual Feasibility: $`\lambda \ge 0`$
**When all 4 conditions above are valid, then KKT will be met**

**KKT Stationarity**:
The general formula of KKT Stationarity condition is as below:
```math
\begin{aligned}
& 0 \in \partial f(\theta) + \lambda\partial g(\theta)\\
& 0 \in \frac{1}{n}X^T(y-X\theta) + \lambda\cdot sign(\theta), sign(0) \in [-1, 1]\\
\end{aligned}
```

In KKT Stationarity, it states that the gradient of MSE is cancelled out by the gradient of the Lasso Penalty. This means both side are of equal value. Thus, we can rewrite this as:
```math
\begin{aligned}
& \frac{1}{n}X^T(y-X\theta) = \lambda\cdot sign(\theta), sign(0) \in [-1, 1]\\
& \text{When }\theta=0\text{, }\\
& \frac{1}{n}X^T(y-X\theta) \in \lambda\cdot [-1, 1]\\
& \frac{1}{n}X^T(y-X\theta) \in [-\lambda, \lambda]\\
&\approx |\frac{1}{n}X^T(y-X\theta)| \le \lambda\\
\end{aligned}
```
Thus, if the gradient of MSE in Lasso is smaller than the hyperparameter, $`\lambda`$, it satisfy KKT Stationarity, which forces the weights to be 0.

**KKT Complementary Slackness**
- It is used to determine the activeness of the budget constraint.
- This is because when the Lagrange Multiplier, $`\lambda`$ is larger than 0, the sum of weights, $`||\theta||_1`$ must be equal to the budget constraint, t, in order for the value to be 0.
- And when both values are equal, it indicates that the budget constraint value is not too large which is able to fit the weights "tightly".

**KKT Primal Feasibility**
- Besides that, in Primal Feasibility, it requires the value of the weight, $`||\theta||_1`$ to be less than the budget constraint t. This is to ensure that the sum of the weights will not exceed the cap limit, t

**KKT Dual Feasibility**
- Based on the Dual Feasibility, it states that $`\lambda \ge 0`$, this means that our Regularization Hyperparameter could not be in negative value.

# Reference
1. Stanford Lagrangian Duality for Dummies: https://www-cs.stanford.edu/people/davidknowles/lagrangian_duality.pdf
2. Stanford CS229 Lagrange Duality Notes
3. CMU Primal and Dual Problem Notes (Lecture 11: Convex Optimization 10-725): https://www.stat.cmu.edu/~ryantibs/convexopt-F15/scribes/11-dual-gen-scribed.pdf
4. CMU KKT Notes (Lecture 12: Convex Optimization 10-725): https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/12-kkt-scribed.pdf
