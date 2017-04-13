import smo as smo
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
    ones = np.ones((1797, 1))
    input = np.hstack((ones, x))
    (a, b) = smo.smo(input, y, 0, 1., 1/2, 1e-4)
    print("Koef1: ")
    print(a)
    return a

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
'''clf = svm()
a = test()
total = 0
for i in range (10000):
    teg = np.random.rand(1, 64)
    yGood = clf.predict(teg)
    teg = np.c_[1, teg]
    res = teg.dot(a)
    if res > 0:
        yMy = 1
    else:
        yMy = -1
    if yGood == yMy:
        total += 1
    #print("Good: " + str(yGood) + ", my: " + str(yMy) + ', val: ' + str(res))
print(total)
print('My coef: ' + str(a))
print('Good coef: ' + str(clf.coef_))'''