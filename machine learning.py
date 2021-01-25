from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

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



##print(model.predict(data_X[:4, :]))
##print(data_y[:4])

'''
X, y = datasets.make_regression(n_samples=100, n_features=2, n_targets=1, noise=2)

plt.scatter(X[:,:1], y , color ='red')
plt.scatter(X[:,1:2], y)
plt.show()
'''