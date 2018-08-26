
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

house.remove_outliers()
house.clean()
house.convert_types(HOUSE_CONFIG)

house.add_features()
house.box_cox()

house.ordinal_features(HOUSE_CONFIG)
house.one_hot_features()

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

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9))

KRR = KernelRidge()

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber')

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread = -1)

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
# %%







house.corr_matrix(house.train(), 'SalePrice', 20)
house.price_corr_cols.drop('SalePrice').values

from sklearn import linear_model
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
plt.rcParams["figure.figsize"] = (12,16)
bsmt_data = house.bx_train[house.price_corr_cols.drop('SalePrice').values]

alpha_100 = np.logspace(0, 8, 100)
coef = []
for i in alpha_100:
    enet.set_params(alpha = i)
    ridge.fit(bsmt_data, house.by_train)
    coef.append(ridge.coef_)

alphas_elastic = np.logspace(-4, 4, 1000)
coef = []

for i in alphas_elastic:
    elastic = linear_model.ElasticNet(l1_ratio =0.5)
    elastic.set_params(alpha = i)
    elastic.fit(bsmt_data, house.by_train)
    coef.append(elastic.coef_)

df_coef = pd.DataFrame(coef, index=alphas_elastic, columns=bsmt_data.columns)
import matplotlib.pyplot as plt
title = 'Ridge coefficients as a function of the regularization'

plot = df_coef.plot(logx=True, title=title)
fig = plot.get_figure()
fig.savefig("output.png")








averaged_models.fit(house.bx_train.values, house.by_train)
prediction = averaged_models.predict(house.bx_test)

submission = pd.DataFrame()
submission['Id'] = house.bx_test_ids
submission['SalePrice'] = np.expm1(prediction)
submission.to_csv('submission.csv',index=False)
submission


import matplotlib.pyplot as plt
var = 'Neighborhood'
data = pd.concat([house.all['LotFrontage'], house.all[var]], axis=1)
sns.mpl.rc("figure", figsize=(14,6))
sns.boxplot(x=var, y="LotFrontage", data=data)
