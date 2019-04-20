import numpy as np


# sigmoid当做激活函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


def loss(ytrue, ypred):
    return ((ytrue - ypred)**2).mean()


class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedforward(self, x):
        # 都是1xn的矩阵， 吧np.dot理解为优化后更好理解
        total = np.dot(self.w, x) + self.b
        return sigmoid(total)


weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)
final = n.feedforward(np.array([2, 3]))
print(final)
print('---------------')


class OurNeuronNet:
    def __init__(self):
        w = np.array([0, 1])
        b = 0
        self.h1 = Neuron(w, b)
        self.h2 = Neuron(w, b)
        self.o1 = Neuron(w, b)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


nn = OurNeuronNet()
print(nn.feedforward(np.array([2, 3])))