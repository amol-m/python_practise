Original file is located at
    https://colab.research.google.com/drive/1sBaoD99c6Irku4aAOQ0hytM39-_a9nYR
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

for col in data:
    print(col)
    print(data[col].unique())
    print('\n')

df = data
df['num_of_doors'].replace('?','four', inplace=True)

df.replace(to_replace=['?'],value = None, inplace=True)

df['normalized_losses'] = df.normalized_losses.str.replace('?', '0')
df['normalized_losses'] = pd.to_numeric(df['normalized_losses'] )
df['normalized_losses'].replace(to_replace=0,value = df['normalized_losses'].mean(), inplace=True)

cont_feats = [col for col in df.columns if df[col].dtype != object]
print(cont_feats)
categorical_feats = [col for col in df.columns if df[col].dtype == object]
df_cont_feats = df[cont_feats]
df_categorical_feats = df[categorical_feats]
print(categorical_feats)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
imputed = imputer.fit_transform(df_cont_feats)
df_cont_feats_imputed = pd.DataFrame(imputed, columns=cont_feats)

df_cont_feats_imputed.head()

df[['bore', 'stroke','horsepower', 'peak_rpm', 'price']] = df[['bore', 'stroke','horsepower', 'peak_rpm', 'price']].apply(pd.to_numeric)

df.describe()

df.columns

df.head()

replace_values ={'four':4,'six':6, 'five':5, 'eight':8,'two':2,'three':3,'twelve':12}           
df = df.replace({"num_of_cylinders": replace_values}) 
df[['num_of_cylinders']] = df[['num_of_cylinders']].apply(pd.to_numeric)

df['num_of_cylinders']

X = df.drop('price' ,axis= 1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

!pip install catboost
import catboost as cb
X_train.columns

X_train.head()

X_train['engine_type']

import catboost as cb
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import seaborn as sns
#!pip install shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
cat_features_index = [2,3,4,5,6,7,8,14,17]

train_dataset = cb.Pool(X_train, y_train,cat_features = cat_features_index) 
test_dataset = cb.Pool(X_test, y_test,cat_features = cat_features_index)

model = cb.CatBoostRegressor(loss_function='RMSE')

grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print('Testing performance')
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))

import matplotlib.pyplot as plt


sorted_feature_importance = model.feature_importances_.argsort()
plt.barh(df.columns[sorted_feature_importance], 
        model.feature_importances_[sorted_feature_importance], 
        color='turquoise')
plt.xlabel("CatBoost Feature Importance")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names = df.columns[sorted_feature_importance])
