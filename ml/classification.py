from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
print(iris.DESCR)
feature = iris.data
label = iris.target
print(feature[0], label[0])
clf = KNeighborsClassifier()
clf.fit(feature, label)
preds = clf.predict([[31, 1, 1, 1]])
print(preds)