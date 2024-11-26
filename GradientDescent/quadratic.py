import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数及其梯度
def f(x, gamma):
    return 0.5 * (x[0]**2 + gamma * x[1]**2)

def grad_f(x, gamma):
    res = [x[0],gamma * x[1]]
    return res

def gradient_descent(Max_iter,gamma):
    y_plot = []
    epsilon = 1e-8
    x = [gamma,1]
    y = f(x,gamma)
    y_plot.append(y)
    t = 2 / (1 + gamma)
    iteration_count = 0
    converge_time = Max_iter
    while(iteration_count < Max_iter):
        gradient = grad_f(x,gamma)
        for i in range(len(gradient)):
            gradient[i] = gradient[i] * t

        for i in range(len(gradient)):
            x[i] = x[i] - gradient[i]

        y = f(x,gamma)
        iteration_count += 1
        if(y < epsilon):
            converge_time = iteration_count
            break
        y_plot.append(y)


    return y_plot, converge_time

if __name__ == '__main__':
    plt.figure(1,figsize=(20, 5))
    gamma_list = [0.04,0.2,1]
    max_iter = 5000
    color = ['green', 'red', 'yellow']
    legend = []
    for i in range(len(gamma_list)):
        y,converge_time = gradient_descent(max_iter,gamma_list[i])
        print('gamma = ',gamma_list[i],end='    ')
        print('converge time = ',converge_time)
        legend.append('gamma =' + str(gamma_list[i]))
        plt.plot(y, color = color[i], marker = 'o',linestyle = '-')
    plt.xlabel("Iter time")
    plt.ylabel("f(x)")
    plt.legend(legend)
    plt.title("The convergence of different gamma")
    plt.show()

    plt.figure(2, figsize=(20, 5))
    gamma_list = [5,25,100]
    color = ['blue', 'black', 'fuchsia']
    legend = []
    for i in range(len(gamma_list)):
        y,converge_time = gradient_descent(max_iter, gamma_list[i])
        print('gamma = ',gamma_list[i],end='    ')
        print('converge time = ',converge_time)
        legend.append('gamma =' + str(gamma_list[i]))
        plt.plot(y, color=color[i], marker='o', linestyle='-')
    plt.xlabel("Iter time")
    plt.ylabel("f(x)")
    plt.legend(legend)
    plt.title("The convergence of different gamma")
    plt.show()


