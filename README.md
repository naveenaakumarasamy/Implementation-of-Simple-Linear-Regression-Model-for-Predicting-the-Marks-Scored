# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries (e.g., pandas, numpy,matplotlib).

2.Load the dataset and then split the dataset into training and testing sets using sklearn library.

3.Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).

4.Use the trained model to predict marks based on study hours in the test dataset.

5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEENAA A K
RegisterNumber: 212222230094 

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head(10)
```
![image](https://github.com/user-attachments/assets/5d67dc13-4b37-477a-ac78-18d44728ed7b)
```

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
```
![image](https://github.com/user-attachments/assets/ed7bb20a-bf46-460f-886f-16f104b29227)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
![image](https://github.com/user-attachments/assets/5c95c320-d6d5-4c5e-b292-d9ec34726ce7)

```
y_pred = reg.predict(X_test)
y_pred
```
![image](https://github.com/user-attachments/assets/18e966f9-f379-4f65-9ef4-1f4de72c9efc)
```
Y_test
```
![image](https://github.com/user-attachments/assets/b1ea9755-109d-4f6c-8cf0-9666d315fd7a)
```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/7939eae2-b37d-495d-8460-7b5b0423dc3a)
```
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,reg.predict(X_train),color='yellow')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![image](https://github.com/user-attachments/assets/7e36edd7-becd-472b-ada5-483540fe0d45)

```
mse = mean_squared_error(Y_test,y_pred)
print("MSE : ",mean_squared_error(Y_test,y_pred))
print("MAE : ",mean_absolute_error(Y_test,y_pred))
print("RMSE : ",np.sqrt(mse))
```
![image](https://github.com/user-attachments/assets/2cb3fc92-ed72-4019-8f74-2dad077c8a43)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
