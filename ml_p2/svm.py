# reading and writing data
import pandas as pd
fruits = pd.read_csv('fruit_data_with_colors.txt', sep='\t')
fruits.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# partition the data into two classes
y_train_1 = y_train == 1  # mandarin in True class, others in False class
y_test_1 = y_test == 1   # mandarin in True class, others in False class
y_train = 2 - y_train_1  # mandarin = 1; others =2
y_test = 2 - y_test_1

seeData = True
if seeData:
    # plotting a scatter matrix
    from matplotlib import cm
    from pandas.plotting import scatter_matrix
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

    # plotting a 3D scatter plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d   # must keep
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('color_score')
    plt.show()

# Create classifier object: kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=100, random_state=10, tol=1e-4)
lsvc.fit(X_train, y_train)
print("Linear SVM Training set score: {:.2f}%".format(100*lsvc.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100*lsvc.score(X_test, y_test)))
#
lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default=â€™rbfâ€™ = radial basis function
# if poly, default degree = 3
from sklearn.svm import SVC
svc = SVC(degree=2, kernel='poly', random_state=1, gamma='auto')
svc.fit(X_train, y_train)
print("SVM Poly Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Poly Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default=â€™rbfâ€™ = radial basis function
from sklearn.svm import SVC
svc = SVC(C=10, gamma='auto', random_state=100)
svc.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# SVM for multiple classes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2)

# SVM with linear kernel
from sklearn.svm import SVC
svc = SVC(C=10, degree=1, kernel='poly')
svc.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))