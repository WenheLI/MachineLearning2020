# Assignment 4
Wenhe Li wl1508

## Problem 1

- 1 
  - a
    - Naive Nayes:
    $$
        P(Y = 1 | X_1, ..., X_p ) = \frac{P(Y=1) \prod_{i=1}^p P(X_i|Y=1)} {\sum_{j=0}^1P(Y=j) \prod_{i=1}^p P(X_i|Y=1)} 
    $$
    - Logistic Regression:
    $$
        P(Y = 1 | X_1, ..., X_p ) = \frac{1}{1 + exp(w_0 + \sum_{i=1}^p w_iX_i)}
    $$
  - b
    - Naive Bayes
    $$
        ln\frac{P(Y=1)}{P(Y=0)} + \sum_{i=1}^p ln \frac{P(x_i|Y=1)}{P(x_i|Y=0)} 
    $$
    - Logistic Regression:
    $$
        w_0 + \sum_{i=1}^p w_iX_i
    $$
  - c
    Naive Bayes: 2n
    Logistic Regression: n
  - d
    Naive Bayes: 2n + 2
    Logistic Regression: 2n

- 2
    Assume, $P(X_j = 1 | Y = 1) = \theta_j$ and $P(X_i = 1 | Y = 0) = \theta_i$,
    we can rewrite the classification for naive bayes by:
    $$
        ln\frac{P(Y=1)}{P(Y=0)} + \sum_{n=1}^p ln \frac{\theta^n_j(1-\theta_j)^{1-n}}{\theta^n_i(1-\theta_i)^{1-n}} \rArr w_0 + \sum_{n=1}^p n (ln\frac{\theta_j}{\theta_i}) + \sum_{i=1}^p (1-n)(ln\frac{1-\theta_j}{1-\theta_i}) \\
        \rArr w_0 +  \sum_{n=1}^p w_nX_n - \sum_{n=1}^p m_n(X_n - 1)
    $$

## Problem 2
- 1
  - a
    $$
      \frac{e^x}{1+e^x} = \frac{1}{\frac{1}{e^x} + 1} = \frac{1}{e^{-x}+1}
    $$ 
  - b
    $$
      \frac{e^{x_c}}{\sum_{c' = 1}^C e^{x_{c'}}} \rArr  \frac{e^{x_c} * e^\delta}{e^\delta * \sum_{c' = 1}^C e^{x_{c'}}} \rArr  \frac{e^{x_c + \delta}}{\sum_{c' = 1}^C e^{x_{c'} + \delta}}
    $$
  - c
    
  - d
- 2
  - a
    - 1 By using log, we can avoid underflow from happening, if the likelihood is extremly small.
    - 2 Also, in the calculation, we might encounter `exp`, which could lead to a overflow. By using log, can avoid it as well.
  - b
    By applying a log on our function, we does not change the `monotonicity` of the function. Therefore, we can still get the classification surface.
- 3
  - a
    $$
    l(W) = ln (P(y=C|x, V) \prod_{c'=1}^{C-1}P(y=c'|x, V)) \rArr ln P(y=C|x, V) + \sum_{c'=1}^{C-1}P(y=c'|x, V)) \\
    \rArr l(W) = -ln(1 + \sum_{c'=1}^{C-1}e^{v^T_{c'}x}) + \sum_{c'=1}^{C-1}（v^T_{c'}x -ln(1 + \sum_{c'=1}^{C-1}e^{v^T_{c'}x}) ）
    $$
  - b