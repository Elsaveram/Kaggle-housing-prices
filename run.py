
## Data loading
# %%
%reload_ext autoreload
%autoreload 2
import math
import json
from house import *
import pandas as pd
from config import *
del house
house = House('data/train.csv','data/test.csv')
# %%

#%%
# Clean/impute data
#house.cleanRP()
house.remove_outliers()
house.clean()
# Convert types after clenaing and filling in all NAs
house.convert_types(HOUSE_CONFIG)
house.add_features()
house.box_cox()
house.all
##Feature Engeneneering:
house.ordinal_features(HOUSE_CONFIG)
#house.label_encode()
house.one_hot_features()

#%%

house.sk_random_forest(house.encoded_all, 100)

pd.set_option('display.max_columns', 500)
house.encoded_all

#Save processed data frames
house.dummy_train.to_csv('dummy_clean_Rachel')

##EDA:
#All variables:
house.all.shape
house.all.head()
house.all.dtypes

house.test().shape
house.train().shape

#Response varriable:
house.train().SalePrice.describe()
house.log_transform(house.train().SalePrice)
house.corr_matrix(house.train(), 'SalePrice')

# Show Missing values:
house.missing_stats()

# Show how data is distributed
house.distribution_charts()

# How does each of our variables relate to sale price?
house.sale_price_charts()

# Understand the Lot Frontage/Area/Config relationship
house.relation_stats('LotFrontage', 'LotArea', 'LotConfig')

# TBD

for category in [x for x in house.all.columns if house.all[x].dtype == 'object']:
    print("Category " + category + " has n unique values " + str(house.all[category].nunique() / house.all.shape[0] * 100) + "%" )


def distskew(dataset, feature):
    fig = plt.figure()
    sns.distplot(dataset[feature], fit=norm);
    return("Skewness = ",skew(dataset[feature].dropna()))

#for c in [x for x in house.bx_train.columns if house.bx_train[x].dtype == 'object' ]:
#    lbl = LabelEncoder()
#    lbl.fit(list(house.bx_train[c].values))
#    print(c)
#    house.bx_train[c] = lbl.transform(list(house.bx_train[c].values))

# shape
print('Shape house.bx_train: {}'.format(house.bx_train.shape))

house.bx_train.sample(10)


##MODELING

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import numpy as np
import pandas as pd



#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True).get_n_splits(house.bx_train.values)
    rmse= np.sqrt(-cross_val_score(model, house.bx_train.values, house.by_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge()

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# %%

house.by_train = np.log1p(house.by_train)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# %%

lasso.fit(house.bx_train.values, house.by_train)
prediction = lasso.predict(house.bx_test)

ENet.fit(house.bx_train.values, house.by_train)
prediction = ENet.predict(house.bx_test)

GBoost.fit(house.bx_train.values, house.by_train)
prediction = GBoost.predict(house.bx_test)

model_xgb.fit(house.bx_train.values, house.by_train)
prediction = model_xgb.predict(house.bx_test)

averaged_models.fit(house.bx_train.values, house.by_train)
prediction = averaged_models.predict(house.bx_test)

submission = pd.DataFrame()
submission['Id'] = house.bx_test_ids
submission['SalePrice'] = np.expm1(prediction)
submission.to_csv('submission.csv',index=False)
submission


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
     def __init__(self, models):
         self.models = models

     # we define clones of the original models to fit the data in
     def fit(self, X, y):
         self.models_ = [clone(x) for x in self.models]

         # Train cloned base models
         for model in self.models_:
             model.fit(X, y)

         return self

     #Now we do the predictions for cloned models and average them
     def predict(self, X):
         predictions = np.column_stack([
             model.predict(X) for model in self.models_
         ])
         return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models = (ENet, GBoost, lasso ))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
