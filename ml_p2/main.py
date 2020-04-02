import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import train_test_split

# In wdbc.data, in the second column labeled 'Diagnosis', a 0 represents 
# benign and a 1 represents malignant

print("Preliminary data: \n")

data = pd.read_csv('wdbc.data')

X = data[['a','b','c','d','e','f','g','h','i','j','0','1','2','3','4','5',
          '6','7','8','9','10','11','12','13','14','15','16','17','18','19']]
y = data['Diagnosis']

#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=.25, random_state=rd.randrange(0, 0x7fffffff))

# Small function to count the number malignant cases in the dataset
malig = 0
for i in y:
   if(i == 1):
      malig += 1

print("\tDatapoint count: {}".format(len(y)))
print("\t Atribute count: {}".format(len(X.columns)))
print("\tMalignant cases: {}".format(malig))
print("\t   Benign cases: {}".format(len(y) - malig))
print("\t Training split: 75%")
print("\t  Testing split: 25%")

visualize = False
if visualize:
   # Generate plots of the data in reduced dimensions
   fig = plt.figure()
   ax = fig.add_subplot(111, projection = '3d')
   ax.scatter(X_train['a'], X_train['b'], X_train['c'], c = y_train, marker = 'o', s=100)
   ax.set_xlabel('Radius')
   ax.set_ylabel('Texture')
   ax.set_zlabel('Perimiter')
   plt.title("Figure 1")

   fig = plt.figure()
   ax = fig.add_subplot(111, projection = '3d')
   ax.scatter(X_train['a'], X_train['g'], X_train['e'], c = y_train, marker = 'o', s=100)
   ax.set_xlabel('Radius')
   ax.set_ylabel('Concavity')
   ax.set_zlabel('Smoothness')
   plt.title("Figure 2")
   plt.show()

input("\nPress enter to run logistic regression:")
print("\nRunning logistic regression:\n")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=rd.randrange(0, 0x7fffffff))
lr.fit(X_train, y_train)

#########################################
# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = 
# b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

sm.plot_confusion_matrix(lr, X_test, y_test, display_labels=['Benign', 'Malignant'], 
                         cmap=plt.cm.Blues)
plt.title("Logistic Regression Test Data Confusion Matrix")
plt.show()

input("\nLogistic regression is finished.\nPress enter to continue.")

###########################################################################################

print("\nRunning Support Vector Machine:\n")

from sklearn.svm import LinearSVC
# partition the data into two classes
y_train_1 = y_train == 1
y_test_1 = y_test == 1
y_train = 2 - y_train_1 
y_test = 2 - y_test_1

# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1
lsvc = LinearSVC(C=100, random_state=rd.randrange(0, 0x7fffffff), tol=1e-4)
lsvc.fit(X_train, y_train)

print("Linear SVM Training set score: {:.2f}%".format(100*lsvc.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100*lsvc.score(X_test, y_test)))

lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)

sm.plot_confusion_matrix(lsvc, X_test, y_test, display_labels=['Benign', 'Malignant'], 
                         cmap=plt.cm.Blues)
plt.title("Support Vector Machine Test Data Confusion Matrix")
plt.show()

input("\nSupport Vector Machine is finished.\nPress enter to continue.")

###########################################################################################

print("\nRunning K Nearest Neighbors with K = 15:\n")

# Create classifier object: kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

sm.plot_confusion_matrix(knn, X_test, y_test, display_labels=['Benign', 'Malignant'], 
                         cmap=plt.cm.Blues)
plt.title("KNN Test Data Confusion Matrix")
plt.show()