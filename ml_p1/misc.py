
import pandas as pd
import numpy as np
import statistics as st
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# In wdbc.data, in the seconds column labeled 'Diagnosis', a 0 represents 
# benign and a 1 represents malignant
data = pd.read_csv('wdbc.data')


X = data[['a','b','c','d','e','f','g','h','i','j','0','1','2','3','4','5',
          '6','7','8','9','10','11','12','13','14','15','16','17','18','19']]
y = data['Diagnosis']

# Open a file to store the data collected

# analysis = open("analysis.txt", "w")
# analysis.write("Max,Min,Mean,Median,Mode,StdDev\n")
# 
# for i in X:
#    analysis.write(str(max(X[i])) + "," + str(min(X[i])) + "," + 
#                   str(st.mean(X[i])) + "," + str(st.median(X[i])) + "," +
#                   str(st.stdev(X[i])) + "\n")


X_train, X_test, y_train, y_test = train_test_split(X, y)
knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
knn.fit(X_train, y_train)
plt.axis

sm.plot_confusion_matrix(knn, X_test, y_test, display_labels=['Benign', 'Malignant'], 
                         cmap=plt.cm.Blues)
plt.show()

j = 0
for i in list(y_test):
   if i == 0:
      j += 1