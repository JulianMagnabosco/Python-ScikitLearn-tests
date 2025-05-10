import numpy as np
from sklearn import datasets, linear_model,model_selection,metrics,neighbors
import matplotlib.pyplot as plt
import pandas as pd  # doctest: +SKIP
import numpy as np

boston = datasets.load_breast_cancer()
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

x=boston.data[:,np.newaxis,3]
# print(x)
y=boston.target
# print(y)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

lr = neighbors.KNeighborsClassifier(n_neighbors=5,metric='minkowski')
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

print("----Presicion----")
print(metrics.accuracy_score(y_test,y_predict))
print("----Exactitud----")
print(metrics.recall_score(y_test,y_predict))
print("----F1----")
print(metrics.f1_score(y_test,y_predict))
print("----ROC-AUC----")
print(metrics.roc_auc_score(y_test,y_predict))
print("----Matriz de confusion----")
print(metrics.confusion_matrix(y_test,y_predict))

# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_predict,color="red",linewidth=3)
# plt.xlabel("Datos x")
# plt.ylabel("Datos y")
# plt.title("Resultado")
# plt.show()
