import numpy as np
from sklearn import datasets, linear_model,model_selection,tree
import matplotlib.pyplot as plt

boston = datasets.load_diabetes()
# print("Keys")
# print(boston.keys())
# print()
# print("DESCR")
# print(boston.DESCR)
# print()
# print("Data Shape")
# print(boston.data.shape)
# print()
# print("Data Shape")
# print(boston.feature_names)

x=boston.data[:,np.newaxis,2]
# print(x)
y=boston.target
# print(y)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

lr = tree.DecisionTreeRegressor(max_depth=3)
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

print("----Presicion----")
print(lr.score(x_train,y_train))

x_flat_test = np.ravel(x_test)
x_grid=np.arange(min(x_flat_test),max(x_flat_test),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x_test,y_test)
plt.plot(x_grid,lr.predict(x_grid),color="red",linewidth=3)
plt.xlabel("Datos x")
plt.ylabel("Datos y")
plt.title("Resultado")
plt.show()

