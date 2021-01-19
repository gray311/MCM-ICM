from pulp import *

'''线性规划'''

# 1. 建立问题
prob = LpProblem("Problem", LpMaximize) #LpMaximize/LpMinimize 求最大值/最小值

# 2. 建立变量
x1 = LpVariable("x1", 0, None, LpContinuous)
x2 = LpVariable("x2", 0, None, LpContinuous)
x3 = LpVariable("x3", 0, None, LpContinuous) #一般是连续型 ，如果需要整数规划的话用LpInteger

'''pulp.LpVariable(name, lowBound=None, upBound=None, cat='Continuous', e=None)
其函数参数为name：变量名；lowBound变量下界；upBound变量上界；cat变量类型，可以为LpInteger\LpBinary\LpContinuous三者之一；e指明变量是否在目标函数和约束中存在，主要用来实现列生成算法。'''

# 3. 设置目标函数
prob += 2 * x1 + 3 * x2 - 5 * x3

# 4. 施加约束
prob += x1 + x2 + x3 == 7
prob += 2 * x1 - 5 * x2 + x3 >= 10.0
prob += 1 * x1 + 3 * x2 + x3 <= 12.0

# 5. 求解
prob.solve() 

# 6. 打印求解状态
print("Status:", LpStatus[prob.status])

# 7. 打印出每个变量的最优值
for v in prob.variables():                                     
    print(v.name, "=", v.varValue)

# 8. 打印最优解的目标函数值
print(prob.name, "=", value(prob.objective))



'''非线性规划'''

from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize

# 1.目标函数
def objective(x): return x[0]** 2 + x[1]** 2 + x[2]** 2 + 8

# 2.约束条件
def constraint1(x): return x[0] ** 2 - x[1] + x[2]**2  # 不等约束 默认形式位>=0

def constraint2(x): return -(x[0] + x[1]**2 + x[2]**2-20)  # 不等约束

def constraint3(x): return -x[0] - x[1]**2 + 2  #等式约束 默认形式为 ==0

def constraint4(x): return x[1] + 2*x[2]**2 -3  #等式约束 默认形式为 ==0       

if __name__ == "__main__":
# 3.边界约束
    b = (0.0, None)
    bnds = (b, b ,b) #都大于0

    con1 = {'type': 'ineq', 'fun': constraint1} #不等式约束 ineq
    con2 = {'type': 'ineq', 'fun': constraint2} #等式约束 eq
    con3 = {'type': 'eq', 'fun': constraint3}
    con4 = {'type': 'eq', 'fun': constraint4}
    cons = ([con1, con2, con3,con4])  # 4个约束条件

# 4.计算
    x0 = np.array([0, 0, 0])
    solution = minimize(objective, x0, method='SLSQP', \
                    bounds=bnds, constraints=cons)  #利用minimize库求解 /求最大值加个负号就行了
    x = solution.x

# 8. 打印最优解的目标函数值
    print('目标值: ' + str(objective(x)))

# 7. 打印出每个变量的最优值
    print('答案为')
    print('x1 = ' + str(x[0]))
    print('x2 = ' + str(x[1]))
    print('x3 = ' + str(x[2]))

