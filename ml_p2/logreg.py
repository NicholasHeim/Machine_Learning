#import pandas as pd
################################
## dataset: Probability of passing an exam versus hours of study
################################
## Constructing DataFrame from a dictionary
#d = {'hours': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 
#               2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5],
#     'pass': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]}
#HrsStudying = pd.DataFrame(data=d)
#
#import matplotlib.pyplot as plt
#plt.plot(HrsStudying["hours"], HrsStudying["pass"], 'ro')
#plt.ylabel('pass')
#plt.xlabel('hours of studying')
#plt.show()
#
#X = HrsStudying[["hours"]] # need [[]], required to be a 2D-array
#y = HrsStudying["pass"]
#
##################################
#from sklearn.model_selection import train_test_split
## random_state: set seed for random# generator
## test_size: default 25% testing, 75% training
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
# random_state=42)
#
## Create a linear model : LogisticRegression
## Requires X be two-dimensional array, 2D array
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(random_state=0)
#lr.fit(X_train, y_train)
#
##########################################
## Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = 
## b0+b_1x_1+b_2x_2+...+b_px_p
#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))
#
## Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#
## Use the trained logistic repression model to predict a new, previously unseen object
#studyHours = 1
#pass_prediction = lr.predict([[studyHours]])
#pass_probability = lr.predict_proba([[studyHours]])
#print("pass: {}".format(pass_prediction[0]))
#print("fail/pass probability: {}".format(pass_probability[0]))

########################################################################################
# Breast Cancer Diagnosis
########################################################################################
import pandas as pd

data = pd.read_csv('wdbc.data')

X = data[['a','b','c','d','e','f','g','h','i','j','0','1','2','3','4','5',
          '6','7','8','9','10','11','12','13','14','15','16','17','18','19']]
y = data['Diagnosis']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=40)
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

## Section following this is not used, this is obsolete for the code above. This is not
## organized to run on the breast cancer data set. 
#
## Use the trained logistic regression classifier model to classify new, previously
## unseen #objects
## first example: a small fruit with mass 20g, color_score = 5.5, width 4.3 cm, 
## height 5.5 cm
##fruit_prediction = lr.predict([[5.5, 4.3, 20, 5.5]])
##print(fruit_prediction[0])

## second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm, #color_score 6.3
##fruit_prediction = lr.predict([[8.5, 6.3, 100, 6.3]])
##print(fruit_prediction[0])
