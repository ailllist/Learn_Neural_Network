import numpy as np
import matplotlib.pyplot as plt

THETA = 0.4
GT = np.array([0.7, 0.2])
# GT = np.random.random(2)
LEARNING_RATE = 0.4
EPOCH = 20

INPUT_VALUE = np.array([0.1, 0.5])

def sigmoid(val):
    return max(THETA, 1 / (1 + np.e ** -val))


def grad_sigmoid(val):
    return val * (1 - val)


class NN:
    
    def __init__(self, input_val, weights1, weights2):
        self.input_val = input_val
        func_sig = np.vectorize(sigmoid)
        self.s11 = input_val @ weights1
        self.s12 = func_sig(self.s11)
        self.s21 = self.s12 @ weights2
        
    def back_propagation(self):
        grad_MSE = self.s21 - GT
        func_grad = np.vectorize(grad_sigmoid)
        grad_weight = self.s12
        u1 = grad_MSE
        u1 = np.expand_dims(u1, axis=0)
        grad_weight = np.expand_dims(grad_weight, axis=1)
        n_weights2 = weights2 - LEARNING_RATE * (grad_weight @ u1)
        
        print("u:\n\b", u1)
        print("w:\n\b", weights2.T)
        
        pu2 = u1 @ weights2.T
        grad_sig1 = func_grad(self.s12)
        u2 = pu2 * grad_sig1
        
        grad_weight = self.input_val
        grad_weight = np.expand_dims(grad_weight, axis=1)
        n_weights1 = weights1 - LEARNING_RATE * (grad_weight @ u2)
        
        return n_weights1, n_weights2


# bias?
weights1 = np.random.random((2, 3))
weights2 = np.random.random((3, 2))
print("GT\n\b", GT)
print("weights1\n\b", weights1)
print("weights2\n\b", weights2)

plot_list = []
loss_list = []

for i in range(EPOCH):
    nn = NN(INPUT_VALUE, weights1, weights2)
    weights1, weights2 = nn.back_propagation()
    print(nn.s21)
    plot_list.append(nn.s21)
    loss_list.append([abs(GT[0]-nn.s21[0]), abs(GT[1]-nn.s21[1])])
    
print("weights1\n\b", weights1)
print("weights2\n\b", weights2)

plot_list = np.array(plot_list)
plt.figure(1)
plt.plot(plot_list[:, 0], "-b")
plt.plot(plot_list[:, 1], "-g")
plt.plot([GT[0] for i in range(EPOCH)], "-r")
plt.plot([GT[1] for i in range(EPOCH)], "-r")
plt.legend(["output1", "output2", "GT"])
plt.title("Epoch: %d (non-Linear classifier)" % EPOCH)

plt.xlabel("Epoch")
plt.ylabel("value")

plt.figure(2)
loss_list = np.array(loss_list)
plt.plot(loss_list[:, 0], "-b")
plt.plot(loss_list[:, 1], "-g")
plt.plot([0 for i in range(EPOCH)], "-k")
plt.legend(["output1", "output2"])
plt.title("train loss graph")

plt.show()
