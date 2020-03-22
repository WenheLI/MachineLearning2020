import numpy as np


# Tune hyper-parameters here.
opts = {
    'threshold': 100000000,
    'num_epochs': 100000000,
    'batch_size': 100000000,
    'init_weight_scale': 10000000000.000000000000001,
    'learning_rate': 10000000000.000000000000001
}


class LinearLayerForward:
    def __call__(self, weights, xs, ctx=None):
        """
        Implement a batched version of linear transformation.
        """

        # Put your code here.
        logits = xs * np.transpose(weights)

        if ctx is not None:
            # Put your code here.
            pass

        return logits


class LinearLayerBackward:
    def __call__(self, ctx, dlogits):
        """
        Get the derivative of the weight vector.
        """
        
        # Put your code here.

        return dw


class LinearLayerUpdate:
    def __call__(self, weights, dw, learning_rate=1.0):
        """
        Update the weight vector.
        """

        # Put your code here.

        return new_weights


class SigmoidCrossEntropyForward:
    def __call__(self, logits, ys, ctx=None):
        """
        Implement a batched version of sigmoid cross entropy function.
        """
        def theta(logits):
            a = np.exp(logits)/(1 + np.exp(logits))
            b = 1/(1 + np.exp(-logits))
            return a if np.isnan(b).any() else b
        # Put your code here.
        ys = ys.astype(int)
        ls = theta(logits)
        print(logits)
        print(ls)
        average_loss = - ys * np.log(ls) - (1 - ys) * np.log(1 - ls)
        average_loss = average_loss.mean()
        print(average_loss)
        if ctx is not None:
            # Put your code here.
            pass

        return average_loss


class SigmoidCrossEntropyBackward:
    def __call__(self, ctx, dloss):
        """
        Get the derivative of logits.
        """
        
        # Put your code here.

        return dlogits


class Prediction:
    def __call__(self, logits):
        """
        Make email classification.
        """

        # Put your code here.

        return predictions
