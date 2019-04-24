import numpy as np


# sigmoid当做激活函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


def mse_loss(ytrue, ypred):
    return ((ytrue - ypred)**2).mean()


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedforward(self, x):
        # 都是1xn的矩阵， 吧np.dot理解为优化后更好理解
        total = np.dot(self.w, x) + self.b
        return sigmoid(total)


class OurNeuronNet:
    def __init__(self):
        # 不同的权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 不同的偏执权重
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] +self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_true):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):
                # 前向传播，过会需要这个值
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1
                d_L_d_ypred = -2*(y_true-y_pred)

                # nauron o1 的他w
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #neuron h2
                d_h1_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h1_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h1_d_b2 = deriv_sigmoid(sum_h2)

                # 更新w和b
                # neuron h1
                self.w1 -=learn_rate*d_L_d_ypred*d_ypred_d_h1*d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -=learn_rate*d_L_d_ypred*d_ypred_d_h1*d_h1_d_b1
                # neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w4
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_b2

                # neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                if epoch % 10==0:
                    y_preds = np.apply_along_axis(self.feedforward,1, data)
                    loss = mse_loss(all_y_true, y_preds)
                    print('epoch %d loss %.3f' %(epoch, loss))


data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6]
])
# 0代表男
all_y_true = np.array([
    1,
    0,
    0,
    1
])
# 创建神经网络，并训练，反向传播
network = OurNeuronNet()
network.train(data, all_y_true)
# 插入数据预测，接近0为男 0.94,0.04
emily = np.array([-7, -3])
frank = np.array([20, 2])
print("emily: %.3f", network.feedforward(emily))
print("frank: %.3f", network.feedforward(frank))