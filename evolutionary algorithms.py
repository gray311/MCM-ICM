'''进化类算法'''

'''

1.差分进化类算法

min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5

DE函数参数：
func	-	目标函数
n_dim	-	目标函数的维度
size_pop	50	种群规模
max_iter	200	最大迭代次数
prob_mut	0.001	变异概率
F	0.5	变异系数
lb	-1	每个参数的最小值
ub	1	每个参数的最大值
constraint_eq	空元组	线性约束
constraint_ueq	空元组	非线性约束
    
'''

def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2

constraint_eq = [
    lambda x: 1 - x[1] - x[2]
]

constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]

from sko.DE import DE

de = DE(func=obj_func, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

'''
模拟退火算法SA:

入参	默认值	意义
func	-	目标函数
x0	-	迭代初始点
T_max	100	最大温度
T_min	1e-7	最小温度
L	300	链长
max_stay_counter

'''

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2

from sko.SA import SA

sa = SA(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y', best_y)

import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
plt.show()


'''

用遗传算法来进行数据拟合
一般先要看出数据大概符合


'''
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

x_true = np.array((1.0, 2.0, 3.0, 4.0))     #随机生成序列，通常会用题目所给段数据来训练
y_true = np.array((1.0, 4.0, 9.0, 16.0))

plt.plot(x_true, y_true, 'o')

def f_fun(x, a, b, c):
    return a * x ** 2 + b * x + c #用二次函数来拟合


def obj_fun(p):
    a, b, c = p
    residuals = np.square(f_fun(x_true, a, b, c) - y_true).sum()
    return residuals

ga = GA(func=obj_fun, n_dim=3, size_pop=100, max_iter=500,lb=[-2] * 3, ub=[2] * 3) 

best_params, residuals = ga.run()
print('best_x:', best_params, '\n', 'best_y:', residuals)

y_predict = f_fun(x_true, *best_params)

x = np.array((5.0, 6.0))

print(f_fun(x, *best_params))

fig, ax = plt.subplots()

ax.plot(x_true, y_true, 'o')
ax.plot(x_true, y_predict, '-')

plt.show()