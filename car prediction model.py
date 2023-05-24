#import part....

import numpy as np
import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


dataset=pd.read_csv("D:\Music\projects'\data science\car prediction model\car_data.csv") #read complete csv file
m=dataset.head() #head for 1st 5 rows
#print(m)

null=dataset["selling_price"].isnull().sum() #sum of null elements in particular column

#value_count..count of elements in each category

fuel_cnt=dataset["fuel"].value_counts() 
own_cnt=dataset["owner"].value_counts()
#print(own_cnt)


#iloc..separting specific row and column
#input and output categories for prediction

X = dataset.iloc[:, [1,4,6,3]].values #input:YOM,fuel,transmission,km driven
Y = dataset.iloc[:, 2].values #output:selling price
#print(X)
#print(Y)


#giving an label value ...1,2,3..... for fuel and transmiision

lb = LabelEncoder()
X[:,1]=lb.fit_transform(X[:,1]) #label for fuel
lb1 = LabelEncoder()
X[:,2]=lb1.fit_transform(X[:,2]) #label for transmission
#print(X)


#train_test split used to separate training set and test set at test size as 0.05

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05, random_state = 0)
#print(X_train[:,:]


#regressor is for estimating continuous numerical values based on given input features

regressor = RandomForestRegressor(n_estimators=300,random_state=0)
reg=regressor.fit(X_train,y_train)
#print(reg)


#accuracy of prediction...from test set

accuracy = regressor.score(X_test,y_test)
#print(accuracy*100,'%')


#sample perediction
#details required for prediction:year of manufacturing,fuel type,transmission type,km driven

yr=int(input("enter the year of manufacturing:"))
fl=input("enter the fuel type:")
tm=input("enter the transmission type:")
km_d=int(input("enter the km driven:"))

pre_model=[yr,fl,tm,km_d]
pre_model[1]=lb.transform([pre_model[1]])[0] #setting the value as label value
pre_model[2]=lb1.transform([pre_model[2]])[0] #setting the value as label value
predicted_price= regressor.predict([pre_model]) 
print("predicted selling price of car is ",predicted_price[0])