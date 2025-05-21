import numpy as np
from sklearn import datasets, linear_model,model_selection,preprocessing
import matplotlib.pyplot as plt

boston = datasets.load_diabetes()
# print("Keys")
# print(boston.keys())
# print()
# print("DESCR")
# print(boston.DESCR)
# print()
# print("Data Shape")
print(boston.data.shape)
# print()
# print("Data Shape")
# print(boston.feature_names)

x=boston.data[:,np.newaxis,3]
# print(x)
y=boston.target
# print(y)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

poli_grade = preprocessing.PolynomialFeatures(degree=2)
x_train_p=poli_grade.fit_transform(x_train)
x_test_p=poli_grade.fit_transform(x_test)

lr = linear_model.LinearRegression()
lr.fit(x_train_p,y_train)

y_predict = lr.predict(x_test_p)

print("----y=a1.x^2 + a2.x +b----\n")
print("----Pendiente (a)----")
print(lr.coef_)
print("----Intersección (b)----")
print(lr.intercept_)
print("----Presicion----")
print(lr.score(x_train_p,y_train))

plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,color="red",linewidth=3)
plt.xlabel("Datos x")
plt.ylabel("Datos y")
plt.title("Resultado")
plt.show()

