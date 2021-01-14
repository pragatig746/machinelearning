import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("day.csv")

data.head()

data.isna().sum()#for checking if data is null 

data.describe()#summary

sns.pairplot(data)

data_1=data[['season','holiday','weekday','workingday','weathersit','temp','windspeed','casual','registered','cnt']]

sns.pairplot(data_1)

data_2=data[['temp','windspeed','casual','registered','cnt']]

sns.pairplot(data_2)

data_2.corr()

feature=data_2['registered'].values
target=data_2['cnt'].values

sns.scatterplot(y=feature,x=target)

from sklearn.linear_model import LinearRegression
linear_mod=LinearRegression()

feature=feature.reshape(-1,1)
target=target.reshape(-1,1)

linear_mod.fit(feature,target)

x_lim=np.linspace(min(feature),max(feature)).reshape(-1,1)
plt.scatter(feature,target)
plt.xlabel('cnt')
plt.ylabel('registered')
plt.title('cnt vs registered')
plt.plot(x_lim,linear_mod.predict(x_lim),color='red')
plt.show()

data_2.columns

x=data_2[['temp', 'windspeed', 'casual', 'registered']]
y=data_2['cnt']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
linear_mod.fit(x_train,y_train)

print(linear_mod.intercept_)

linear_mod.coef_

predict=linear_mod.predict(x_test)
predict

y_test

plt.scatter(y_test,predict)

sns.distplot(y_test-predict)
