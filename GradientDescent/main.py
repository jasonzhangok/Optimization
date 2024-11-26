import matplotlib.pyplot as plt
from math import exp, log10
import numpy as np
from scipy.optimize import minimize_scalar


def Func(x1, x2):
    res = exp(x1 + 3 * x2 - 0.1) + exp(x1 - 3 * x2 - 0.1) + exp(-x1 - 0.1)
    return res


def grad_func(x1, x2):
    res = [exp(x1 + 3 * x2 - 0.1) + exp(x1 - 3 * x2 - 0.1) - exp(-x1 - 0.1),
           3 * exp(x1 + 3 * x2 - 0.1) - 3 * exp(x1 - 3 * x2 - 0.1)]
    return res


def backtracking(alpha, beta):  # 回溯直线搜索
    x1 = x2 = 0.5
    y = Func(x1, x2)
    maxIter = 300

    curve2 = [y]
    iterationcount = 0
    gradient = grad_func(x1, x2)
    epsilon = 1e-5
    while (gradient[0] ** 2 + gradient[1] ** 2 > epsilon * epsilon and iterationcount < maxIter):
        iterationcount += 1
        gradient = grad_func(x1, x2)
        # 找t_k
        t_k = 1.0
        while (Func(x1 - t_k * gradient[0], x2 - t_k * gradient[1]) >
               y - alpha * t_k * (gradient[0] ** 2 + gradient[1] ** 2)):
            t_k *= beta
        x1 = x1 - t_k * gradient[0]
        x2 = x2 - t_k * gradient[1]
        y_iter = Func(x1, x2)
        y = y_iter

        curve2.append(y)
    y_star = y
    # calculate distance between points form curve2 to y*
    for i in range(len(curve2)):
        curve2[i] -= y_star
    return curve2, y_star, iterationcount


def backtracking_show():  # 回溯直线搜索绘图
    print("Backtracking:")
    # alpha change
    plt.figure(1, figsize=(20, 5))
    index = 0
    color = ['green', 'red', 'yellow', 'blue', 'black', 'orange', 'fuchsia', 'indigo', 'navy', 'gray']
    legend = []
    while (index <= 4):
        alpha = (index + 1) / 10
        curve, y, k = backtracking(alpha, 0.7)
        for i in range(len(curve) - 1):
            curve[i] = -1 * log10(curve[i])
        curve.pop()
        plt.plot(curve, color=color[index], marker='o')
        print("alpha = ", alpha, end='   ')
        print("result = ", y, end='  ')
        print("iteration time = ", k)
        legend.append("alpha = " + str(alpha))
        index += 1
    plt.xlabel("Iter K")
    plt.ylabel("-log(f(x)-p*)")
    plt.legend(legend)
    plt.title("beta = 0.7, goal function value with alpha change")
    plt.show()

    # beta change
    plt.figure(2, figsize=(20, 5))
    index = 0
    legend = []
    color = ['green', 'red', 'yellow', 'blue', 'black', 'orange', 'fuchsia', 'indigo', 'navy', 'gray']
    while (index <= 8):
        beta = (index + 1) / 10
        curve, y, k = backtracking(0.1, beta)
        for i in range(len(curve) - 1):
            curve[i] = -1 * log10(curve[i])
        curve.pop()
        plt.plot(curve, color=color[index], marker='o')
        print("beta = ", beta, end='   ')
        print("result = ", y, end='  ')
        print("iteration time = ", k)
        legend.append("beta = " + str(beta))
        index += 1
    plt.xlabel("Iter K")
    plt.ylabel("-log(f(x)-p*)")
    plt.legend(legend)
    plt.title("alpha = 0.1, goal function value with beta change")
    plt.show()
    return


def ExactLineSearch():  # 精确直线搜索
    x1 = x2 = 0.5
    y = Func(x1, x2)
    maxIter = 30

    curve = [y]
    iterationcount = 0
    while (iterationcount < maxIter):
        iterationcount += 1
        gradient = grad_func(x1, x2)

        def line_search(alpha):
            return Func(x1 - alpha * gradient[0], x2 - alpha * gradient[1])

        res = minimize_scalar(line_search)
        alpha_opt = res.x  # optimal t

        x1 = x1 - alpha_opt * gradient[0]
        x2 = x2 - alpha_opt * gradient[1]

        cur_res = Func(x1, x2)
        curve.append(cur_res)
    return curve, cur_res


def ExactLineSearch_show():
    plt.figure(figsize=(20, 20))
    curve, y = ExactLineSearch()
    plt.plot(curve, color='red', marker='o')
    print("ExactLineSearch result = ", y, end='  ')
    plt.xlabel("Iter K")
    plt.ylabel("f(x)")
    # plt.title("f(x) with change of iterate times")
    plt.show()


backtracking_show()
ExactLineSearch_show()
