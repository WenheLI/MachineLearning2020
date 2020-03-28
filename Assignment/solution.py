import numpy as np


# Tune hyper-parameters here.
opts = {
    'threshold': 29,
    'num_epochs': 250,
    'batch_size': 100,
    'init_weight_scale': 0.009,
    'learning_rate': .95 
    }

# 30 100 50 .5 .01
# 30 100 50 .5 .5
# 30 200 100 .5 .75 .05% 2.4%
# 25 200 100 .5 .75 .05% 3.5%
# 25 200 125 .5 .75 .05% 3.6%
# 25 200 125 .5 .85 .05% 3.2%
# 25 300 100 .5 .75 .025% 1.9%
# 25 300 150 .5 .75 .025% 3.5%
# 25 300 300 .5 .75 .075% 2.7%
# 25 300 125 .5 .5 .075% 2.8%
# 27 300 100 .1 .75 0% 2.5%
# 27 300 100 .01 .75 0% 2.4%
# 27 300 100 .01 .85 0.025% 3.2%
# 27 300 300 .01 .85 0.05% 2.7%
# 27 300 500 .01 .99 0.05% 2.6%
# 27 300 250 .0005 .95 0.05% 3.1%
# 27 350 150 .0001 .9 0% 2.3%
# 29 250 100 .009 .95 0% 2.3% 3.8%

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
            ctx['ys'] = np.asarray(dys) / len(logits)
            ctx['exp'] = np.asarray(dexp) / len(logits)

        return average_loss


class SigmoidCrossEntropyBackward:
    def __call__(self, ctx, dloss):
        """
        Get the derivative of logits.
        """
        
        # Put your code here.
        dlogits = ctx['ys'] + ctx['exp']
        return dlogits


class Prediction:
    def __call__(self, logits):
        """
        Make email classification.
        """

        # Put your code here.
        predictions = logits > 0
        return predictions
