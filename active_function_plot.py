import math
import matplotlib.pyplot as plt

x_range=list(range(-10,11))

def sigmoid(x_range):
    sig_y=[]
    for i in range(len(x_range)):
        sig_y.append(1/(1+pow(math.e,-x_range[i])))
    plt.figure()
    plt.plot(x_range,sig_y)
    plt.vlines(0, 0, 1, colors = "c", linestyles = "dashed")
    plt.hlines(0, -10, 10, colors = "c", linestyles = "dashed")
    plt.title('sigmoid function')
    plt.xlabel('input signal')
    plt.ylabel('output sigmal')
    plt.show()

def tanh(x_range):
    tanh_y=[]
    for i in range(len(x_range)):
        tanh_y.append((pow(math.e,x_range[i])-pow(math.e,-x_range[i]))/(pow(math.e,x_range[i])+pow(math.e,-x_range[i])))
    plt.figure()
    plt.plot(x_range,tanh_y)
    plt.vlines(0, -1, 1, colors = "c", linestyles = "dashed")
    plt.hlines(0, -10, 10, colors = "c", linestyles = "dashed")
    plt.title('tanh function')
    plt.xlabel('input signal')
    plt.ylabel('output sigmal')
    plt.show()
    
def relu(x_range):
    relu_y=[]
    for i in range(len(x_range)):
        if x_range[i]>=0:
            relu_y.append(x_range[i])
        if x_range[i]<0:
            relu_y.append(0)
    plt.figure()
    plt.plot(x_range,relu_y)
    plt.vlines(0, -10, 10, colors = "c", linestyles = "dashed")
    plt.hlines(0, -10, 10, colors = "c", linestyles = "dashed")
    plt.title('relu function')
    plt.xlabel('input signal')
    plt.ylabel('output sigmal')
    plt.show()
    

sigmoid(x_range)
tanh(x_range)
relu(x_range)