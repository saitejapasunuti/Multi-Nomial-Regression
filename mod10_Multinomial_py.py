import pandas as pd
#pandas is used for data manipulation,analysis and cleansing
import numpy as np
#numpy is used for numerical data
import seaborn as sns
#statistical data vizualization library
from sklearn.model_selection import train_test_split
#it is used for splitting the data into train and test subsets
 from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#used to calculate accuracy score of the model

student=pd.read_csv('D:\\360digiTMG\supervised\mod 10 multinomial Regression\mdata.csv')
student

#remove the index and id column
student=student.drop(student.columns[[0,1]],axis=1)

student.head()#gives first 5 rows of the dataset
student.head(10)#gives first 10 rows of the dataset
student.tail(10)#gives last 10 rows of the dataset
student.describe()#gives 1st moment decision values

student.prog.value_counts()
#academic    105
#vocation     50
#general      45

#boxplot of independent varible distribution of each category of prog
sns.boxplot(x="prog",y="read",data=student)
sns.boxplot(x="prog",y="write",data=student)
sns.boxplot(x="prog",y="math",data=student)
sns.boxplot(x="prog",y="science",data=student)

#scatter plot
sns.stripplot(x="prog",y="read",jitter=True,data=student)
sns.stripplot(x="prog",y="write",jitter=True,data=student)
sns.stripplot(x="prog",y="math",jitter=True,data=student)
sns.stripplot(x="prog",y="science",jitter=True,data=student)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(student,hue="prog")
sns.pairplot(student)#normal

#correlation values between each inndependent varibale
student.corr()

#split the data into train and test subsets
train_data,test_data=train_test_split(student,test_size=0.2)

#logistic regression supports only with 'lgbfs' and 'newton-cg','liblinear','sag', 'saga' solvers
#multinomial is supported only with 'lgbfs' and 'newton-cg' solvers
model=LogisticRegression(multi_class='multinomial',solver='lbfgs').fit(train_data.iloc[:,5:8],train_data.iloc[:,0])

train_pred=model.predict(train_data.iloc[:,5:8])
test_pred=model.predict(test_data.iloc[:,5:8])

#train accuracy
accuracy_score(train_data.iloc[:,0],train_pred)
# 0.6875

#test accuracy
accuracy_score(test_data.iloc[:,0],test_pred)
#0.675
