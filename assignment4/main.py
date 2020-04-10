import time
import math
import numpy as np
class NeuralNetwork:
    def __init__(self, ws, bs, ws2, bs2):
        super().__init__()
        self.ws = ws
        self.bs = bs
        self.ws2 = ws2
        self.bs2 = bs2
        self.res = []

    def logistic(self, x):
        if x > 0:
            return 1/(1+math.exp(-x))
        else:
            e = math.exp(x)
            return e/(1+e)

    def hidden_unit(self, x, w, b):
        temp = 0
        for idx in range(len(x)):
            temp += x[idx] * w[idx]
        return self.logistic(temp+b)

    def softmax(self, z):
        temp = list(map(lambda value: math.exp(value), z))
        _sum = sum(temp)
        return list(map(lambda value: value/_sum, z))

    def output_layer(self, x, w, b):
        return self.softmax(list(map(lambda value, idx: self.hidden_unit(x, value, b[idx]), w, range(len(w)))))

    def forward(self, xs):
        return self.output_layer(list(map(lambda w, idx: self.hidden_unit(xs, w, bs[idx]), self.ws, range(len(self.ws)))), self.ws2, self.bs2)

    def classify(self, x):
        res = self.forward(x)
        self.res.append(res)
        return res.index(max(res))

    def digit_classifier(self, xs, ys):
        res = list(map(lambda x, idx: self.classify(x) == (ys[idx]-1), xs, range(len(xs))))
        pos = len(list(filter(lambda x: x, res)))
        return 1 - pos/len(res)


class NeuralNetworkNP:
    def __init__(self, ws, bs, ws2, bs2, is_relu=False):
        super().__init__()
        self.ws = ws
        self.bs = bs
        self.ws2 = ws2
        self.bs2 = bs2
        self.res = []
        self.is_relu = is_relu

    def logistic(self, x):
        if x > 0:
            return 1/(1+np.exp(-x))
        else:
            e = np.exp(x)
            return e/(1+e)

    def hidden_unit(self, x, w, b):

        temp = x.dot(w) + b
        if self.is_relu:
            return np.max(temp, 0)
        else:
            return 1/(1+np.exp(-temp))

    def softmax(self, z):
        temp = np.exp(z)
        _sum = temp.sum()
        return z/_sum

    def output_layer(self, x, w, b):
        t = np.asarray(list(map(lambda value, idx: self.hidden_unit(x, value, b[idx]), w, range(len(w)))))
        return self.softmax(t)

    def forward(self, xs):
        t = np.asarray(list(map(lambda w, idx: self.hidden_unit(xs, w, bs[idx]), self.ws, range(len(self.ws)))))
        return self.output_layer(t, self.ws2, self.bs2)

    def classify(self, x):
        res = self.forward(x)
        self.res.append(res)
        return res.argmax()

    def digit_classifier(self, xs, ys):
        res = list(map(lambda x, idx: self.classify(x) == (ys[idx]-1), xs, range(len(xs))))
        pos = len(list(filter(lambda x: x, res)))
        return 1 - pos/len(res)


def vscHelper(input):
    return list(map(lambda x: float(x),input.strip().split(",")))

if __name__ == "__main__":

    with open("./ps5_data-labels.csv", "r") as f:
        labels = list(map(lambda x: int(x.strip()) ,f.readlines()))

    with open("./ps5_data.csv", "r") as f:
        inputs = list(map(lambda x: vscHelper(x) ,f.readlines()))

    with open("./ps5_theta1.csv", "r") as f:
        res = f.readlines()
        matrix = list(map(lambda x: vscHelper(x), res))
        bs = [m[0] for m in matrix]
        ws = [[m[i] for i in range(1, len(matrix[0]))] for m in matrix]

    with open("./ps5_theta2.csv", "r") as f:
        res = f.readlines()
        matrix = list(map(lambda x: vscHelper(x), res))
        bs2 = [m[0] for m in matrix]
        ws2 = [[m[i] for i in range(1, len(matrix[0]))] for m in matrix]

    # startTime = time.time()

    # net = NeuralNetwork(ws, bs, ws2, bs2)
    # val = net.digit_classifier(inputs, labels)
    
    # print(time.time() - startTime)
    # print(val)

    ws_np = np.asarray(ws)
    bs_np = np.asarray(bs)
    bs2_np = np.asarray(bs2)
    ws2_np = np.asarray(ws2)
    inputs_np = np.asarray(inputs)
    labels_np = np.asarray(labels)

    startTime = time.time()
    npnet = NeuralNetworkNP(ws_np, bs_np, ws2_np, bs2_np)    

    val = npnet.digit_classifier(inputs_np, labels_np)
    
    print(time.time() - startTime)
    print(val)

    startTime = time.time()
    npnet = NeuralNetworkNP(ws_np[:20], bs_np[:20], ws2_np[:,:20], bs2_np[:20])    

    val = npnet.digit_classifier(inputs_np, labels_np)
    
    print(time.time() - startTime)
    print(val)

    startTime = time.time()
    npnet = NeuralNetworkNP(ws_np, bs_np, ws2_np, bs2_np, is_relu=True)    

    val = npnet.digit_classifier(inputs_np, labels_np)
    
    print(time.time() - startTime)
    print(val)

    startTime = time.time()
    npnet = NeuralNetworkNP(ws_np[:20], bs_np[:20], ws2_np[:,:20], bs2_np[:20], is_relu=True)    

    val = npnet.digit_classifier(inputs_np, labels_np)
    
    print(time.time() - startTime)
    print(val)