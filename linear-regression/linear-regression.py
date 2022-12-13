"""
Original file is located at
    https://colab.research.google.com/drive/1dXyyqZ5F8PfNXtq6z5YNp240FKDBVO2M
"""

import pandas as pd

column_name = [
'symboling',
'normalized_losses',
'make',
'fuel_type',
'aspiration',
'num_of_doors',
'body_style',
'drive_wheels',
'engine_location',
'wheel_base',
'length',
'width',
'height',
'curb_weight',
'engine_type',
'num_of_cylinders',
'engine_size',
'fuel_system',
'bore',
'stroke',
'compression_ratio',
'horsepower',
'peak_rpm',
'city_mpg',
'highway_mpg',
'price'
]

data = pd.read_csv("automobile.csv", names=column_name , header=None)

data.head(10)

for col in data:
    print(col)
    print(data[col].unique())
    print('\n')

data.info()

data.shape

data['num_of_doors'].value_counts()

df = data
df['num_of_doors'].replace('?','four', inplace=True)

df['num_of_doors'].value_counts()

# df.replace(to_replace= ['?'],value = None, inplace=True)
df.replace(to_replace=['?'],value = None, inplace=True)

df['normalized_losses'] = df.normalized_losses.str.replace('?', '0')

df['normalized_losses'].unique()

df['normalized_losses'] = pd.to_numeric(df['normalized_losses'] )
df['normalized_losses'].replace(to_replace=0,value = df['normalized_losses'].mean(), inplace=True)

df.info()

pd.set_option('display.max_columns', None)
df.describe()

for col in df:
    print(col)
    print(df[col].unique())
    print('\n')

cont_feats = [col for col in df.columns if df[col].dtype != object]
print(cont_feats)
categorical_feats = [col for col in df.columns if df[col].dtype == object]
df_cont_feats = df[cont_feats]
df_categorical_feats = df[categorical_feats]

df_cont_feats.info()

df_categorical_feats.info()

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
imputed = imputer.fit_transform(df_cont_feats)
df_cont_feats_imputed = pd.DataFrame(imputed, columns=cont_feats)

df_cont_feats_imputed.describe()

for col in df_cont_feats_imputed:
    print(col)
    print(df_cont_feats_imputed[col].unique())
    print('\n')

df_categorical_feats.head()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (OneHotEncoder(sparse=False), ['make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','engine_type','fuel_system']),
    remainder='passthrough')
transformed = transformer.fit_transform(df_categorical_feats)

transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())

print(transformed_df.columns)

df_displacement = transformed_df[['num_of_cylinders', 'bore', 'stroke']]

df_displacement
df_displacement.replace(to_replace=['?'],value = None, inplace=True)

transformed_df[['bore', 'stroke','horsepower', 'peak_rpm', 'price']] = transformed_df[['bore', 'stroke','horsepower', 'peak_rpm', 'price']].apply(pd.to_numeric)

transformed_df.info()

#transformed_df['num_of_cylinders'].value_counts()   
replace_values ={'four':4,'six':6, 'five':5, 'eight':8,'two':2,'three':3,'twelve':12}           
transformed_df = transformed_df.replace({"num_of_cylinders": replace_values})

transformed_df[['num_of_cylinders']] = transformed_df[['num_of_cylinders']].apply(pd.to_numeric)

transformed_df['num_of_cylinders'].value_counts()

final_df = pd.concat([transformed_df, df_cont_feats_imputed], axis=1)

final_df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = final_df.drop('price' ,axis= 1)
y = final_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

regr = LinearRegression()
  
model = regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

from sklearn.metrics import mean_absolute_error,mean_squared_error

y_pred = regr.predict(X_test)
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

print("Model coefficients:\n")
for i in range(X.shape[1]):
    print(X.columns[i], "=", model.coef_[i].round(5))

# Print the Intercept:
print('intercept:', model.intercept_)

# Print the Slope:
print('slope:', model.coef_)

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data
rf_model = regressor.fit(X_train, y_train)

print(rf_model.score(X_test, y_test))

! pip install shap
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)

cont_feats = [col for col in X_train.columns if X_train[col].dtype != object]
print(cont_feats)
categorical_feats = [col for col in X_train.columns if X_train[col].dtype == object]
df_cont_feats = X_train[cont_feats]
df_categorical_feats = X_train[categorical_feats]

print(categorical_feats)

#type(X_train)
#X_train_numpy = X_train.to_numpy
print(y)

! pip install lime
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function

predict_fn_linreg = lambda x: model.predict_proba(x).astype(float)
predict_fn_rf = lambda x: rf_model.predict_proba(x).astype(float)


# Line-up the feature names

feature_names = sum([categorical_feats, cont_feats ], [])
print(feature_names)
feature_names_cat = list(categorical_feats) 

# Create the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['price'], verbose=True, mode='regression')

# Pick the observation in the validation set for which explanation is required
j = 2

# Get the explanation for RandomForest
exp = explainer.explain_instance(X_test.values[j], rf_model.predict, num_features=69)
exp.show_in_notebook(show_all=False)

# Pick the observation in the validation set for which explanation is required
j = 5

# Get the explanation for RandomForest
exp = explainer.explain_instance(X_test.values[j], rf_model.predict, num_features=69)
exp.show_in_notebook(show_all=False)

# Pick the observation in the validation set for which explanation is required
j = 5

# Get the explanation for RandomForest
exp = explainer.explain_instance(X_test.values[j], rf_model.predict, num_features=69)
exp.show_in_notebook(show_all=False)

# decision tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor( max_depth=3, random_state=1234)

# Train Decision Tree Classifer
dt_reg = dt_reg.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt_reg.predict(X_test)

from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt_reg, feature_names=feature_names, filled=True)
