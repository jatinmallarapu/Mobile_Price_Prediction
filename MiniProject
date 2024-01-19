#Project Description: 
#Problem statement:Create a classification model to predict whether 
#price range of mobile based on certain specifications
#Context:An entrepreneur has started his own mobile company. He wants to 
#give tough fight to big companies like Apple, Samsung etc.
#He does not know how to estimate price of mobiles his company creates. In 
#this competitive mobile phone market, one cannot simply assume things. To 
#solve this problem, he collects sales data of mobile phones of various 
#companies. He wants to find out some relation between features of a mobile 
#phone (e.g., RAM, Internal Memory etc) and its selling price. But he is not so 
#good at Machine Learning. So, he needs your help to solve this problem. In this 
#problem you do not have to predict actual price but a price range indicating
#how high the price is
#Dataset link:
https://drive.google.com/file/d/1nZWfzIMYbtc8ZVKwQ9BrM-qYsdN-5L8O/view?usp=sharing


#Code:-
import numpy as np
import pandas as pd
import seaborn as sns
df=pd.read_csv("mobile_price_range_data.csv")
def eval_metrics(y_test, y_pred, plt_title):
 print("Accuracy score ",accuracy_score(y_pred,y_test))
 cm=confusion_matrix(y_test, y_pred)
 print(classification_report(y_test, y_pred))
 sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
 plt.xlabel('Predicted Values')
 plt.ylabel('Actual Values')
 plt.title(plt_title)
 plt.show()
df.head()
#diplay the number of values in the (target or) ‘price_range’ attribute
df['price_range'].value_counts()
#check for null elements
df.isnull().sum()
df.duplicated().sum()
df.shape
sns.countplot(df['price_range'])
df.info()
df.describe()
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,cmap=plt.cm.Accent_r)
plt.show()
df.plot(kind='box',figsize=(20,30))
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train.head()
x_test.head()

#Logistic regression model
from sklearn.linear_model import LogisticRegression
m1=LogisticRegression()
m1.fit(x_train,y_train)
print('Model Training Score ',m1.score(x_train,y_train))
print('Model Testing Score ',m1.score(x_test,y_test))

ypred_m1=m1.predict(x_test)
print(ypred_m1)

from sklearn.metrics import confusion_matrix,classification_report,accu
racy_score
cm=confusion_matrix(y_test,ypred_m1)
print("confusion matrix \n",cm)
print("Accuracy score ",accuracy_score(ypred_m1,y_test))
print("CLASSIFICATION REPORT:- \n",classification_report(y_test,ypred_m
1))

eval_metrics(y_test,ypred_m1,"Logistic regression Confusion Matrix")

m=m1.coef_
c=m1.intercept_
print('Coefficient ',m)
print('Intercept ',c)

import numpy as np
def sig(x,m,c):
 logit=1/(1+np.exp(1-(m*x+c)))
 print(logit)
ypred_values=m1.predict([[19770,1,2.3,1,0,1,31,0.3,151,6,7,900,800,2000
,11,6,15,1,0,1]])
print("The prdicted value for the above features",ypred_values)

#SVM Classifier, kernel = ’linear’

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x=df["battery_power"],y=df["clock_speed"],hue=df["price
_range"])
plt.show()

sns.scatterplot(x=df["battery_power"],y=df["dual_sim"],hue=df["price_ra
nge"])
plt.show()

from sklearn.svm import SVC
m2=SVC(kernel='linear',C=1)
m2.fit(x_train,y_train)
print('Training Score ',m2.score(x_train,y_train))
print('Testing Score ',m2.score(x_test,y_test))

ypred_m2=m2.predict(x_test)
print(ypred_m2)

eval_metrics(y_test,ypred_m2,"SVM Confusion Matrix using kernel='linear
'")

eval_metrics(y_test,ypred_m2,"SVM Confusion Matrix using kernel='linear
'")
#kernel = ’rbf'
#for suppose if we use rbf in svm classifier we get
m3=SVC(kernel='rbf',C=1)
m3.fit(x_train,y_train)
print('Training Score ',m3.score(x_train,y_train))
print('Testing Score ',m3.score(x_test,y_test))
ypred_m3=m3.predict(x_test)
print(ypred_m3)

eval_metrics(y_test,ypred_m3,"SVM Confusion Matrix using 'rbf' kernel")


print(f"R2 score = {metrics.r2_score(y_test, ypred_m3)}")
print(f"Mean Squared Log Error for the Regressor = {metrics.mean_square
d_log_error(y_test, ypred_m3)}")
plt.figure(figsize=(7,5))
sns.regplot(y_test, ypred_m3, fit_reg=True, scatter_kws={"s": 100})
plt.title("SVM Classifier 'rbf' kernel")
plt.show()

#Predicting values using kernel = ‘linear’
#predicting values using svm
x = df
a = np.array(x)
y = a[:,-1] # classes having 0 and 1
 
# extracting two features
x = np.column_stack((x.battery_power,x.blue,x.clock_speed,x.dual_sim,x.
fc,x.four_g,x.int_memory,x.m_dep,x.mobile_wt,x.n_cores,x.pc,x.px_height
,x.px_width,x.ram,x.sc_h,x.sc_w,x.talk_time,x.three_g,x.touch_screen,x.
wifi))
 
# 569 samples and 2 features
x.shape
 
print (x,y)

from sklearn.svm import SVC 
clf = SVC(kernel='linear')
 
# fitting x samples and y classes
clf.fit(x, y)

clf.predict([[777,1,2.3,1,0,1,31,0.3,151,6,7,900,800,2000,11,6,15,1,0,1
]])

clf.predict([[1021,1,0.5,1,0,1,53,0.7,136,3,6,905,1998,2631,18,8,17,1,1
,0]])
clf.predict([[1821,0,1.7,0,4,1,53,0.7,136,3,6,905,1998,2631,18,8,17,1,1
,0]])
clf.predict([[1300,1,0.5,1,0,1,45,0.1,190,3,6,905,890,2631,18,8,17,1,0,
0]])

#Predicting values using kernel = ‘rbf’ 
from sklearn.svm import SVC 
clf_rbf= SVC(kernel='rbf')
 
# fitting x samples and y classes
clf_rbf.fit(x, y)
clf_rbf.predict([[777,1,2.3,1,0,1,31,0.3,151,6,7,900,800,2000,11,6,15,1
,0,1]])
clf.predict([[1031,1,0.5,1,0,1,53,0.7,136,3,6,905,1998,2631,18,8,17,1,1
,0]])
clf.predict([[1721,0,1.7,0,4,1,53,0.7,136,3,6,905,1998,2631,18,8,17,1,1
,0]])
clf.predict([[1901,1,1.4,0,4,1,75,0.9,136,3,6,400,1300,2631,20,8,17,1,0
,0]])

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,leaf_size=25)
knn.fit(x_train, y_train)
y_pred_knn=knn.predict(x_test)
print('KNN Classifier Accuracy Score: ',accuracy_score(y_test,y_pred_kn
n))
cm_rfc=eval_metrics(y_test, y_pred_knn, 'KNN Confusion Matrix')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
def knn_comparison(data, k):
 plt.figure(figsize=(15,10))
 x = data[["battery_power","clock_speed"]].values
 y = data["price_range"].astype(int).values
 clf = neighbors.KNeighborsClassifier(n_neighbors=k)
 clf.fit(x, y)
 # Plotting decision region
 plot_decision_regions(x, y, clf=clf, legend=2)
 # Adding axes annotations
 plt.xlabel("X")
 plt.ylabel("Y")
 plt.title("Knn with K="+ str(k))
 plt.show()
knn_comparison(df,100)

#Accuracy for the three models
print("Accuracy score for Linear Regression model is ",accuracy_score(y
pred_m1,y_test)) #0.642
print("Accuracy score for Linear Regression model is ",accuracy_score(y
pred_m1,y_test)) #0.972
print("Accuracy score for KNN Classifier is ",accuracy_score(y_pred_knn
,y_test)) #0.928

#Report the model with the best accuracy
#From the above accuracy scores of the three models we can say that SVM classifier having kernel ‘linear’ has the highest accuracy score of about 97.2% . 
#So we can say that this model can predict more accurately compared to the other models which we have created in this project.

#Inorder to run above codes please click on the google colab link below:
https://colab.research.google.com/drive/15z-jgMhsYW8xhK0IFZs1KR1EIGQIr1nf?usp=sharing
