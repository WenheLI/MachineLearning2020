import matplotlib.pyplot as plot
import math

def norm_func(x, miu, theta):
    base = 1/((2*math.pi*(theta)) ** .5)
    power = math.exp(-.5 * ((x - miu) ** 2)/theta)
    return base * power
xs = []
ys = []
for x in range(0, 1000, 1):
    temp = x/1000
    xs.append(temp)
    ys.append(norm_func(temp, .5, .01))

plot.plot(xs, ys)
plot.show()