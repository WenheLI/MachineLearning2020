# Neural Network
Wenhe Li wl1508

![](data.png)

Above is the screenshot for `execution time`, `error rate`, and `loss` (for naive py only)
By using numpy, we significantly reduce the `execution time` while doing matrix related operation.

By putting less weights, we make the model contian less information so that it will result in a drop of accuracy.
As for the `ReLu` activation, compared with `softmax`, the parameters are trained upon `softmax` activation. Therefore, using `ReLu` will natrually reduce the accuracy.
That's why `softmax` is better thatn `ReLu` and `Less Weight` will lead to a lower accuracy.