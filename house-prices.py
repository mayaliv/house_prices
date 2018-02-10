import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit

#training_data_file="train.csv"
training_data_file="train.csv"
real_data_file="test.csv"

train_data=pd.read_csv(training_data_file)
real_data=pd.read_csv(real_data_file)
train_data.fillna('',inplace=True)

Numeric = [label for label in train_data.columns if train_data.dtypes[label] != 'object']
Numeric.remove('Id')
Numeric.remove('SalePrice')
Texts = [label for label in train_data.columns if train_data.dtypes[label] == 'object']
Numeric_data = train_data[Numeric]

#replacing 'NA' with 'NaN'
#Numeric = Numeric.applymap(float)

y=train_data['SalePrice']
#X=train_data.remove('SalePrice')
#X=train_data.drop(['SalePrice'], axis = 1)

Texts_data = pd.get_dummies( train_data, columns = Texts )

#print(type(X))
print(type(y))
print(type(Numeric_data))
print(type(Texts_data))
# print(Numeric_data)
# # implementing mean strategy to replace 'NaN'
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit(Numeric_data)
# Numeric_data = imp.transform(Numeric_data)
# Numeric_data = pd.DataFrame(data=Numeric_data[0:,1:], columns=Numeric_data[0,1:])

X = pd.concat([Numeric_data, Texts_data], axis=1)
#print(result)
X=X.drop('SalePrice',axis=1)
X.to_csv('out.csv',index=False)

#Debug: only textual data
#X=Texts_data.drop('SalePrice',axis=1)
#Debug: only Numeric data
X=Numeric_data

print(type(X))






# #
# # #assigning the linear regresion model
reg=LinearRegression()
#
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
cv_scores = cross_val_score(reg,X,y,cv=10, scoring='r2')
#
print(cv_scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
predicted = cross_val_predict(reg, X, y, cv=10)
print(predicted)


#
# # Split using ALL data in sample_df
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=22)
#
#
#
# # Compute and print accuracy
# accuracy = pl.score(X_test, y_test)
# print("\nAccuracy on sample data - all data: ", accuracy)

