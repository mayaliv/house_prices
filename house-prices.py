
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score

#Define the file names
training_data_file="train.csv"
real_data_file="test.csv"

#import the dataset
train_data=pd.read_csv(training_data_file)
real_data=pd.read_csv(real_data_file)

#1. Bedroom number
train_data['BedroomAbvGr'].value_counts().plot(kind='bar')
plt.title('Number of rooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine()

plt.savefig("Bedroom_number.png")
plt.close()


#2. plot salesprice vs sqft
plt.scatter(train_data["SalePrice"], train_data["LotArea"])
plt.title('Price vs SqFt')
plt.xlabel('Price')
plt.ylabel("Area")
#plt.show()
plt.savefig("salesprice_vs_sqft.png")
plt.close()

#3. plot salesprice vs bedrooms
plt.scatter(train_data['BedroomAbvGr'],train_data['SalePrice'])
plt.title('Price_vs_Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel("Price")
#plt.show()
plt.savefig("salesprice_vs_bedrooms.png")
plt.close()

#4. plot salesprice vs Year Built
plt.scatter(train_data['YearBuilt'],train_data['SalePrice'])
plt.title('Price vs Bedrooms')
plt.xlabel('Year Built')
plt.ylabel("Price")
#plt.show()
plt.savefig("salesprice_vs_Year_Built.png")
plt.close()

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

# # Split using to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=22)
#
#
clf = Ridge(alpha=1.0, normalize=True)
results=clf.fit(X_train, y_train)
pred=clf.predict(X_test)
score=clf.score(X_test, y_test)
print('Ridge')
print(score)
#print(pred)
#



modelCV = RidgeCV(alphas = [0.1, 0.01, 0.001,0.0001],cv=5,normalize=True)
modelCV.fit(X_train, y_train)
pred_cv=modelCV.predict(X_test)
scorecv=modelCV.score(X_test, y_test)
mean_squared_error(y_test, pred_cv)

print('RidgeCV')
print(scorecv)
#print(pred_cv)
#


regr = ElasticNetCV(alphas=[0.1, 0.01, 0.001,0.0001], copy_X=True, cv=5, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=1,
       normalize=True, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0)
regr.fit(X_train, y_train)
print('ElasticNetCV')
#print(regr.alpha_)
#print(regr.intercept_)
pred_enet_cv=regr.predict(X_test)
#print(pred_enet_cv)
score_enet_cv=regr.score(X_test, y_test)
print(score_enet_cv)

regr_rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=10,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
regr_rf.fit(X_train, y_train)
pred_rf=regr.predict(X_test)

score_rf=regr_rf.score(X_test, y_test)
print('RandomForestRegressor')
#print(regr_rf.feature_importances_)
print(score_rf)


xgb_model = xgb.XGBRegressor()
xgb_reg = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
xgb_reg.fit(X_train, y_train)
pred_xgb=xgb_reg.best_estimator_.predict(X_test)
print('xgb.XGBRegressor')
print(xgb_reg.best_score_)
print(xgb_reg.best_params_)
xgb_model.fit(X_train,y_train)
#pred_xgb = xgb_model.predict(data=X_test)
print(r2_score(y_test, pred_xgb))