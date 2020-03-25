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

        # def normalize(inputs):
        #     _min = inputs.min()
        #     _max = inputs.max()
        #     temp = _max - _min
        #     if temp == 0:
        #         temp = 1
        #     return (inputs - _min)/temp
        # xs = normalize(xs)
       
        weight_t = np.transpose(weights)
        logits = []
        for x in xs:
            logits.append(np.dot(weight_t, x))
        logits = np.asarray(logits)
        if ctx is not None:
            # Put your code here.
            ctx['weight'] = xs
        return logits


class LinearLayerBackward:
    def __call__(self, ctx, dlogits):
        """
        Get the derivative of the weight vector.
        """

        dl = ctx['weight']
        f = dl.shape[1]
        dw = np.zeros(f)

        for i in range(len(dlogits)):
            dw += np.dot(dlogits[i], dl[i])
        return dw


class LinearLayerUpdate:
    def __call__(self, weights, dw, learning_rate=1.0):
        """
        Update the weight vector.
        """

        # Put your code here.

        new_weights = weights - learning_rate * dw

        return new_weights


class SigmoidCrossEntropyForward:
    def __call__(self, logits, ys, ctx=None):
        """
        Implement a batched version of sigmoid cross entropy function.
        """

        dys = []
        dexp = []

        def theta(logits, ys):
            logits = np.copy(logits)
            for i in range(len(logits)):
                if (logits[i] >= 0):
                    dy = (1 - ys[i])
                    exp = np.exp(-logits[i])
                    logits[i] = dy * logits[i] + np.log(1+exp)
                    dys.append(dy)
                    dexp.append(-exp/(1+exp))
                else:
                    dy = -ys[i]
                    exp = np.exp(logits[i])
                    logits[i] = dy * logits[i] + np.log(1+exp)
                    dys.append(dy)
                    dexp.append(exp/(1+exp))
            return logits
        # Put your code here.
        ys = ys.astype(int)
        average_loss = theta(logits, ys)        
        average_loss = average_loss.mean()

        if ctx is not None:
            # Put your code here.
            ctx['ys'] = np.asarray(dys)
            ctx['exp'] = np.asarray(dexp)

        return average_loss


class SigmoidCrossEntropyBackward:
    def __call__(self, ctx, dloss):
        """
        Get the derivative of logits.
        """
        
        # Put your code here.
        dlogits = ctx['ys'] + ctx['exp']
        print(ctx['ys'].shape)
        print(ctx['exp'].shape)
        print(dlogits.shape)
        return dlogits


class Prediction:
    def __call__(self, logits):
        """
        Make email classification.
        """

        # Put your code here.

        return predictions
