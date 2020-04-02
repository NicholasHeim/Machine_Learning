
import pandas as pd
# In wdbc.data, in the seconds column labeled 'Diagnosis', a 0 represents 
# benign and a 1 represents malignant
data = pd.read_csv('wdbc.data')


X = data[['a','b','c','d','e','f','g','h','i','j','0','1','2','3','4','5',
          '6','7','8','9','10','11','12','13','14','15','16','17','18','19']]
y = data['Diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# plotting a scatter matrix
from matplotlib import cm
from pandas.plotting import scatter_matrix

# plotting a 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
fig = plt.figure()

# Create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15, weights = 'uniform')

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# How sensitive is k-NN classification accuracy to the 
# choice of the 'k' parameter?
k_range = range(1, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

sum = 0
for i in scores:
    sum += i
sum = sum / 50

#plt.figure()
plt.title('K Trials: Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.show()


import numpy as np
t = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 
     0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
for s in t:
    scores = []
    for i in range(1, 1500):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.title('Training v. Testing Trials: Distance')
plt.xlabel('Training set proportion (%)')
plt.ylabel('Accuracy')
plt.show()

# Beyond here are the methods of verifying the kNN machine learning algorithm

import pandas as pd
data = pd.read_csv('wdbc.data')

X = data[['a','b','c','d','e','f','g','h','i','j','0','1','2','3','4','5',
          '6','7','8','9','10','11','12','13','14','15','16','17','18','19']]
y = data['Diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create a linear model : Linear regression (aka ordinary least squares)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_)
# Estimate the accuracy of the classifier on future data, using the test data
# score = 1-relative score
# R^2(y, hat{y}) = 1 - {sum_{i=1}^{n} (y_i - hat{y}_i)^2}/{sum_{i=1}^{n} (y_i - bar{y})^2}
#########################################################################################
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test))

# Malignant = 1, Benign = 0
print("Diagnosis: {}".format(prediction[0]))


##########################################################################################
import pandas as pd

#########################################################################
#########################################################################
#Breast cancer dataset
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
#Classes 2
#Samples per class 212(M),357(B)
#Samples total 569
#Dimensionality 30
#Features real, positive
#########################################################################
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = bc.target

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a linear model : Linear regression (aka ordinary least squares)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("sum lr.coef_^2: {}".format(sum(lr.coef_*lr.coef_)))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation: default 5-fold cross validation cv=5
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr, X, y, cv=5)

# plotting the data
from matplotlib import pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(y, predicted, edgecolors=(0, 0, 0))
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')

# Leave one out: Provides train/test indices to split data in train/test sets. Each
# sample is used once as a test set (singleton) while the remaining samples form the training set.
# n= the number of samples
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
predicted = []
measured = []
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train, y_train)
    predicted.append(lr.predict(X_test)[0])
    measured.append(y_test[0])

ax2.scatter(measured, predicted, edgecolors=(0, 0, 0))
ax2.plot([min(measured), max(measured)], [min(measured), max(measured)], 'k--', lw=4)
ax2.set_xlabel('Measured')
ax2.set_ylabel('Predicted')
plt.show()

###############################################################
# Ridge regression --- a more stable model
# In ridge regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||^2_2 + alpha * ||w||^2_2
# Note: the smaller alpha = the less restriction.
###############################################################
for a in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    print("ridge.coef_: {}".format(ridge.coef_))
    print("sum ridge.coef_^2: {}".format(sum(ridge.coef_*ridge.coef_)))
    print("ridge.intercept_: {}".format(ridge.intercept_))
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

###############################################################
# Lasso regression --- a more stable model
# In lasso regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||_2 + alpha * ||w||_2
# Note: the smaller alpha = the less restriction.
###############################################################
for a in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=a).fit(X_train, y_train)
    print("lasso.coef_: {}".format(lasso.coef_))
    print("sum lasso.coef_^2: {}".format(sum(lasso.coef_*lasso.coef_)))
    print("lasso.intercept_: {}".format(lasso.intercept_))
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))