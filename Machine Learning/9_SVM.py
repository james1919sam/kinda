# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:01:06 2018


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn import svm

style.use("ggplot")

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11],
             [10,10],
             [3,2]])

y = [0,1,0,1,0,1,1,0]

clf = svm.SVC(kernel='linear')

clf.fit(X,y)

print("Prediction is:", clf.predict([[7,4.0]]))

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()