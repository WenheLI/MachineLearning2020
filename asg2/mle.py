from functools import reduce
import matplotlib.pyplot as plot

def mle(D, theta):
    res = 1
    for d in D:
        res *= ((theta ** d) * ((1 - theta) ** (1 - d)))
    return res

if __name__ == "__main__":
    D = [1] * 6 + [0] * 4
    init = 0
    xAxis = []
    yAxis = []
    maxX = 0
    maxY = 0
    while init <= 1.01:
        xAxis.append(init)
        yAxis.append(mle(D, init))
        if yAxis[-1] > maxY:
            maxY = yAxis[-1]
            maxX = xAxis[-1]
        init += .01
    plot.plot(xAxis, yAxis)
    plot.plot([maxX, maxX], [0, maxY])
    plot.show()