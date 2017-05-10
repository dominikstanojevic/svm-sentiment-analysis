import SVM
import numpy as np
import timeit
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits

x = np.random.rand(50000, 100).astype(float)
y = np.random.choice([-1, 1], (50000,), ).astype(int)

digits = load_digits()
x = digits.data
y = digits.target
for i in range(1797):
    y[i] = -1 if y[i] == 0 else 1
print(y)


#x = np.array([[0, 0], [1, 1], [2, 2]]).astype(float)
#y = np.array([-1, 1, 1]).astype(int)

def test():
    svm = SVM.LinearSVM()
    svm.fit(x, y)
    print(svm.get_coef())
    return svm

def svm():
    clf = LinearSVC()
    clf.fit(x, y)
    print("Koef2: ")
    print(clf.coef_)
    return clf


t= timeit.Timer(svm)
print(t.timeit(1))
t = timeit.Timer(test)
print(t.timeit(1))
clf = svm()
a = test()
total = 0
test = np.random.rand(1, 64)
for i in range(9999):
    test = np.vstack((test, np.random.rand(1, 64)))
yGood = clf.predict(test)
yMy = a.predict(test)
print(np.sum(yGood == yMy))
