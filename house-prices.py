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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeCV

#Define the file names
training_data_file="train.csv"
real_data_file="test.csv"

#import the dataset
train_data=pd.read_csv(training_data_file)
real_data=pd.read_csv(real_data_file)

#set y as the target
y=train_data['SalePrice']
train_data=train_data.drop(['Id', 'SalePrice'], axis=1)

#split the data to numeric and catagorical
Numeric = [label for label in train_data.columns if train_data.dtypes[label] != 'object']
Texts = [label for label in train_data.columns if train_data.dtypes[label] == 'object']
Numeric_data = train_data[Numeric]

#preprocess the catagorical data
Texts_data = train_data[Texts]
Texts_data = pd.get_dummies(Texts_data, columns = Texts )
Texts_data.to_csv('text_out1.csv',index=False)

# implementing mean strategy to replace 'NaN'
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_data[Numeric])
features1 = imp.transform(Numeric_data)
Numeric_data = pd.DataFrame(features1, columns=Numeric)

#normelizing the data
features=Numeric_data
scaler = MinMaxScaler().fit(features.values)
#scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X = pd.DataFrame(features, columns = Numeric)
X.to_csv('Num-out.csv',index=False)

#merge text and numeric data
X = pd.concat([X, Texts_data], axis=1)

X.to_csv('out.csv',index=False)
y.to_csv('yout.csv',index=False)

# assigning the linear regresion model
reg=LinearRegression()
#
shuffle = KFold(len(X), n_folds=5, shuffle=True, random_state=0)
#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
cv_scores = cross_val_score(reg,X,y,cv=shuffle)
#
# print(cv_scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
# predicted = cross_val_predict(reg, X, y, cv=shuffle)
# print(predicted)


#
# # Split using to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=22)
#
#
clf = Ridge(alpha=1.0, normalize=True)
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
score=clf.score(X_test, y_test)
print('Ridge')
print(score)
#print(pred)
#


modelCV = RidgeCV(alphas = [0.1, 0.01, 0.001,0.0001],cv=5)
modelCV.fit(X,y)
pred_cv=modelCV.predict(X)
scorecv=modelCV.score(X,y)

print('RidgeCV')
print(scorecv)
#print(pred_cv)
#

