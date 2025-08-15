# Notes for Black-Litterman Class

### Solving for Posterior Expected Returns

The formula is given by the following:

$$
E(R) = [ ( \tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1}\Pi + P^T \Omega^{-1} Q]
$$

To solve this, we first put into the form $\bold{A} x = \bold{b}$, for convenience let $E(R) = \mu$. Then, 
$$
[ ( \tau \Sigma)^{-1} + P^T \Omega^{-1} P] \times \mu = [(\tau \Sigma)^{-1}\Pi + P^T \Omega^{-1} Q]
$$

Using $\bold{A} = ( \tau \Sigma)^{-1} + P^T \Omega^{-1} P$ and $\bold{b} = (\tau \Sigma)^{-1}\Pi + P^T \Omega^{-1} Q$


Then to calculate posterior estimate of the covariance matrix we use the formula

$$
\hat{\Sigma} = \Sigma +[(\tau \Sigma)^{-1} + P^T \Omega^{-1}P]^{-1}
$$

This is easier to solve as we can just use a matrix inverse.


To calculate the weights from these, we do the inverse of caluclating $\Pi$

$$
w = (\delta \Sigma)^{-1}E(R)
$$


