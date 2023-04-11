import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# # to display in notebook
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# read and explore the data
ad_data = pd.read_csv('advertising.csv')

ad_data.head()
ad_data.info()
ad_data.describe()

# check if data is missing usually using a heatmap
sns.heatmap(ad_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# create a jointplot showing Area Income versus Age
sns.jointplot(x='Age',y='Area Income',data=ad_data)

# create a jointplot showing the kde distributions of Daily Time spent on site versus Age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')

# create a jointplot of Daily Time Spent on Site versus Daily Internet Usage
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data,color='green')

# create a pairplot with the hue defined by the Clicked on Ad feature
sns.pairplot(data=ad_data,hue='Clicked on Ad')

# train and test the model and predict values using logistic regression
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))


