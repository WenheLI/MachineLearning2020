# HW1

## Program 1

## Problem 2
Let's take two arbitrary points from the hyperplane $x_1$ amd $x_2$.

Substitute the two points to the hyperplane, we will get:
$$
 w \cdot x_1 + b = 0
$$
$$
 w \cdot x_2 +b = 0 
$$

By doing subtrackion on above equations, we will get:
$$
    w \cdot (x_1 - x_2) = 0
$$
which shows, vector $w$ dot product any line in the plane equals zero and it means vector $w$ is perpendicular to the hyperplane.

## Problem 3
- Claim 2 :

    We have $w_t \rArr w_{t-1} + x_{t-1}y_{t-1}$

    Therefor, $w_t\cdot w^* = (w_{t-1} + x_{t-1}y_{t-1})w^* = w_{t-1}w^* + y_{t-1}x_{t-1}w^*$.

    We know that $y_iw^*x_i \ge \gamma$, then we know $w_tw^* = w_{t-1}w^* + y_{t-1}x_{t-1}w^* \ge w_{t-1}w^* + \gamma$.

    Thus, we know $w_tw^* \ge \gamma + w_{t-1}w^*$; 

    Similarly, we know $w_{t-1}w^* \ge \gamma + w_{t-2}w^*$

    By substituting them iteratively, we will have $w_t\cdot w^* \ge M_t\gamma$

    Claim 2 proof done.

- Claim 3:
    