import numpy as np
import matplotlib.pyplot as plt
import time
# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function with L2 regularization
def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    cost = 1/m * np.sum(np.log(1 + np.exp(-y @ X @ theta)))
    reg_cost = cost + lambda_/(2) * np.sum(theta**2)
    return reg_cost

def time_it(func):
    def inner(*args):
        start = time.perf_counter()
        result = func(*args)
        print(f'total time cost: {time.perf_counter() - start}')
        return result
    return inner

# Define the gradient function with L2 regularization
def gradient_reg(theta, X, y, lambda_):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X @ theta)
    grad = 1/m * X.T @ ((h - y) * X @ theta)
    grad = grad + lambda_/m * theta
    return grad

def data_set(train_size):
    dataset = np.array([[np.random.random()-1*i, np.random.random()-1*i, 2*i-1] for x in range(train_size) for i in range(2)])
    return dataset

data = data_set(train_size=1000)
np.random.shuffle(data)

mask1 = data[:,2]==1
mask0 = ~mask1
data_p = data[mask1][:, :2]
data_n = data[mask0][:, :2]


lambda_ = 1  # Regularization parameter

# Gradient descent
num_iterations = 5000  # Number of iterations
alpha = 0.1  # Learning rate
n = 2 # freature dimention
theta = np.random.random((n, 1))  # Parameters

@time_it
def batch_method(num_iterations):
    global alpha, n, theta, data
    data_bt = data
    theta_bt = theta
    cost_bt = []
    bt_theta_record = []
    X, y = data_bt[:,:2], data_bt[:,2]
    for _ in range(num_iterations):
        grad = gradient_reg(theta_bt, X, y, lambda_)
        theta_bt = theta_bt - alpha * grad
        cost_bt.append(cost_function_reg(theta_bt, X, y, lambda_))
        bt_theta_record.append(theta_bt)
    return cost_bt, np.squeeze(bt_theta_record)

@time_it
def min_sg(num_iterations, batch_size):
    """min_batch sg"""
    global alpha, n, theta, data
    sg_data = data
    sg_theta = theta
    sg_theta_record = []
    cost_sg = []
    for _ in range(num_iterations):
        np.random.shuffle(sg_data)
        sg_temp = sg_data[: batch_size]
        X,y = sg_temp[:,:2], sg_temp[:,2]
        grad = gradient_reg(sg_theta, X, y, lambda_)
        sg_theta = sg_theta - alpha * grad
        sg_theta_record.append(sg_theta)
        cost_sg.append(cost_function_reg(sg_theta, X, y, lambda_))
    return cost_sg, np.squeeze(sg_theta_record)


bt, bt_theta_record = batch_method(num_iterations)
sg40, sg_theta_record = min_sg(num_iterations, 40)
sg10, sg_theta_record = min_sg(num_iterations, 10)
sg100, sg_theta_record = min_sg(num_iterations, 100)



# plt.scatter(data_p[:,0], data_p[:, 1], c='r', alpha= 0.6)
# plt.scatter(data_n[:, 0], data_n[:, 1], c='g', alpha= 0.6)
# sg_theta_record = sg_theta_record/np.linalg.norm(sg_theta_record, axis= -1)[:, np.newaxis]
# bt_theta_record = bt_theta_record/np.linalg.norm(bt_theta_record, axis= -1)[:, np.newaxis]

# for i in range(10):
#     print(i)
#     plt.plot([sg_theta_record[i*20][0], 0], [sg_theta_record[i*20][1], 0], c='r', alpha= 0.1 * (i+1))
#     plt.plot([bt_theta_record[i*20][0], 0], [bt_theta_record[i*20][1], 0], c='g', alpha= 0.1 * (i+1))
# plt.legend(['label 1', 'label -1'])
# plt.show()
x = [i for i in range(len(bt))]
plt.loglog(x, bt, c='r', alpha = 0.6)
plt.loglog(x, sg10, c='g', alpha = 0.6)
plt.loglog(x, sg40, c='b', alpha = 0.6)
plt.loglog(x, sg100, c='y', alpha = 0.6)


plt.xlabel('Iterations (logarithmic scale)')
plt.ylabel('Loss (logarithmic scale)')
plt.legend(['loss_batch2000', 'loss_sg10', 'loss_sg40', 'loss_sg100'])
plt.show()
print(f"The optimized parameters are: {theta}")
