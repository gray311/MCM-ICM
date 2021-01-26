from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # k近邻分类法
from sklearn.linear_model import LinearRegression # 处理线性回归的model
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing  #标准化数据
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC  #处理分类问题的model
from sklearn.model_selection import cross_val_score  #测试数据挑选
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits


iris = datasets.load_iris()

X = iris.data
y = iris.target

'''交叉验证 cross validation'''
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error') # for regression
    ##scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

##print(knn.predict(X_test))
##print(y_test)


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

X, y = datasets.make_regression(n_samples=1000, n_features=1, n_targets=1, noise=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.predict(X_test[:4]))
print(y_test[:4])

X1 = X_test
y1 = model.coef_ * X1 + model.intercept_

plt.plot(X1, y1, color='yellow' )
plt.scatter(X_train, y_train)
#plt.scatter(X_test, y_test,color= 'red')
plt.show()


print(model.coef_)  #y=0.5x+1 输出 0.5 
print(model.intercept_) #输出 1
print(model.get_params()) #给model 定义的参数
print(model.score(X_test,y_test)) #给测试数据打分，看处理效果 ，方法是R^2检验，回归系数


##print(model.predict(data_X[:4, :]))
##print(data_y[:4])

'''
X, y = datasets.make_regression(n_samples=100, n_features=2, n_targets=1, noise=2)

plt.scatter(X[:,:1], y , color ='red')
plt.scatter(X[:,1:2], y)
plt.show()
'''

'''标准化数据'''

a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float64)
print(a)
print(preprocessing.scale(a))

X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
'''
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show() '''

X = preprocessing.scale(X)  #压缩到0，1的范围
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))



'''over fitting'''

digits = load_digits()

X = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.01), X, y, cv=10,
                            scoring='neg_mean_squared_error',
                            train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Crossing-validation')

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range, cv=10,
        scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()