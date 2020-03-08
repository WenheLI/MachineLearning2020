import matplotlib.pyplot as plot

def mapFunc(D, theta):
    res = 1
    for d in D:
        res *= ((theta) ** d) * ((1 - theta) ** (1 - d))
        beta = ((theta ** 2)*(1 - theta) ** 2) / .0333
        return res * beta

if __name__ == "__main__":
    D = [1] * 6 + [0] * 4
    init = 0
    xAxis = []
    yAxis = []
    maxX = 0
    maxY = 0
    while init <= 1.01:
        xAxis.append(init)
        yAxis.append(mapFunc(D, init))
        if yAxis[-1] > maxY:
            maxY = yAxis[-1]
            maxX = xAxis[-1]
        init += .01
    plot.plot(xAxis, yAxis)
    plot.plot([maxX, maxX], [0, maxY])
    plot.show()