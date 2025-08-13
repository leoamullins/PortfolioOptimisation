# Methodology 

This will detail the maths behind the code and models used in this coding project. 

# 1 Maths Framework

## 1.1 Mean-Variance optimisation

This code is based of the mean-variance optimisation and the Markowitz model, which solves for weights *w*. These weights should maximise expected-returns at any given volatility (risk). Mathematically, this can be formulated as

$$
\min_w  \frac{1}{2} w^T  \Sigma  w
$$

Subject to:
- $w^T \mu = \mu_{tr}$ (target return constraint)
- $\mathbf{1}^T w = 1$ (full investment i.e weights sum to 1)
- $w_i \geq 0$ (no short positions)

Where:
- $\Sigma$ is the covariance matrix of asset returns
- $\mu$ is the vector of expected returns
- $\mu_{tr}$ is the target portfolio return

In the textbook, the $\frac{1}{2}$ is added for convenience. Upon further research, the $\frac{1}{2}$ simplifies gradient calculations and helps match up with standard quadratic programming form.

## 1.2 Maximum Sharpe Ratio

The maximum Sharpe ratio problem:

$$\max_{w} \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$

Is non-convex and numerically unstable. Following Cornuejols & Tutuncu (2006), we use the convex reformulation:

$$\min_{x} x^T \Sigma x$$

Subject to:
- $(\mu - r_f \mathbf{1})^T x = 1$
- $\mathbf{1}^T x = \kappa$
- $x_i \geq 0$

This turns it into a concave problem, so there is a global optimum so chances of the optimisation failing are a zero. Since we use a variable transformation, we use $w = \hat{w} / \kappa $. to retrieve the weights.

From the weights, a simple calculation is needed to calculate the sharpe ratio

$$
\text{sharpe} = \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
$$

where $\mu$ is expected returns, $\Sigma$ is the covariance matrix and $r_f$ is the risk-free rate. In this model I defaulted this to 0.