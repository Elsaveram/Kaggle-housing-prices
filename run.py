
# %%
%reload_ext autoreload
%autoreload 2
import math
import json
from house import *
from config import *
del house
house = House('data/train.csv','data/test.csv')
house.remove_outliers()
house.clean(HOUSE_CONFIG)
house.convert_types(HOUSE_CONFIG)
house.ordinal_features(HOUSE_CONFIG)
house.add_features()
house.box_cox()
house.one_hot_features()
#%%

house.bx_train.to_csv('x_train.csv')
house.by_train.to_csv('y_train.csv')
house.bx_test.to_csv('x_test.csv')
pd.set_option('display.max_columns', 500)

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

house.sale_price_charts()

house.all.MasVnrArea.dtype

# shape
print('Shape house.bx_train: {}'.format(house.bx_train.shape))

house.bx_train.sample(10)


##MODELING
# %%
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

import numpy as np
import pandas as pd

#OLS
model_ols = linear_model.LinearRegression()

model_rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

lasso = make_pipeline(RobustScaler(), Lasso(alpha= 0.0006))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha= 0.0006, l1_ratio=.9))

KRR = KernelRidge()

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber')

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,
                            min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
                            subsample=0.5213, silent=1, nthread = -1)

# %%
score = house.rmsle_cv(model_ols)
print("\model_ols score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(model_rf)
print("\model_rf score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = house.rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


ols_prediction = house.fit_and_predict(model_ols)
rf_pred = house.fit_and_predict(model_rf)
lasso_prediction = house.fit_and_predict(lasso)
enet_prediction = house.fit_and_predict(ENet)
gboost_prediction = house.fit_and_predict(GBoost)
# TBD: Feature name mismatch
#xgb_prediction = house.fit_and_predict(model_xgb)


print("beta_1, beta_2: " + str(np.round(model_ols.coef_, 3)))
print("beta_0: " + str(np.round(model_ols.intercept_, 3)))
print("RSS: %.2f" % np.sum((model_ols.predict(house.bx_train) - house.by_train) ** 2))
print("R^2: %.5f" % model_ols.score(house.bx_train, house.by_train))
# %%

from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-5, 2, 30)
grid = GridSearchCV(estimator=Lasso(),param_grid=dict(alpha=alphas), cv=5, scoring='r2')
grid.fit(house.bx_train, house.by_train) # entire datasets were fed here

print(grid.best_params_, grid.best_score_)# score -0.0470788758558
for params, mean_score, scores in grid.grid_scores_:
    print(mean_score, params)

import pandas as pd
import matplotlib
feature_importances = pd.DataFrame(model_rf.feature_importances_,
                                   index = house.bx_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

top_30 = feature_importances.head(30).index.tolist()
top_30
feature_importances.head(30)


data = house.bx_train[top_30]
lasso = Lasso()
alphas_lasso = np.logspace(-4, 2, 100)
coef_lasso = []
for i in alphas_lasso:
    lasso.set_params(alpha = i).fit(data, house.by_train)
    coef_lasso.append(lasso.coef_)

df_coef = pd.DataFrame(coef_lasso, index=alphas_lasso, columns=data.columns)
title = 'Lasso coefficients as a function of the regularization'
matplotlib.rcParams['figure.figsize'] = [12, 16]
fig, ax = plt.subplots( nrows=1, ncols=1 )
ax = df_coef.plot(logx=True, title=title)

fig = ax.get_figure()
fig.savefig('foo.png')


# %% Model Averaging
averaged_models = AveragingModels(models = (ENet, GBoost, lasso))
score = house.rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_prediction = house.fit_and_predict(averaged_models)
# %%


# %% Model Stacking
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR), meta_model = lasso)
score = house.rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# %%



from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#Stacked Regressor

stacked_averaged_models.fit(house.bx_train.values, house.by_train.values)
stacked_train_pred = stacked_averaged_models.predict(house.bx_train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(house.bx_test))
print(rmsle(house.by_train.values, stacked_train_pred))

#Xgboost
model_xgb.fit(house.bx_train.values, house.by_train)
xgb_train_pred = model_xgb.predict(house.bx_train.values)
xgb_pred = np.expm1(model_xgb.predict(house.bx_test.values))
print(rmsle(house.by_train, xgb_train_pred))

#OLS
model_ols.fit(house.bx_train.values, house.by_train)
model_rf_train_pred = model_ols.predict(house.bx_train.values)
model_rf_pred = np.expm1(model_ols.predict(house.bx_test.values))
print(rmsle(house.by_train, model_rf_train_pred))

#Rforest
model_rf.fit(house.bx_train.values, house.by_train)
model_rf_train_pred = model_rf.predict(house.bx_train.values)
model_rf_pred = np.expm1(model_rf.predict(house.bx_test.values))
print(rmsle(house.by_train, model_rf_train_pred))

#DF predictions predictions
ols_df=pd.DataFrame()
ols_df['stacked_pred']=stacked_train_pred
ols_df['xgb_pred']=xgb_train_pred
ols_df['rf_pred']=model_rf_train_pred

#OLS
from sklearn import linear_model
ols = linear_model.LinearRegression()
ols.fit(ols_df, house.by_train)
print("beta_1, beta_2: " + str(np.round(ols.coef_, 3)))
print("beta_0: " + str(np.round(ols.intercept_, 3)))
print("RSS: %.2f" % np.sum((ols.predict(ols_df) - house.by_train) ** 2))
print("R^2: %.5f" % ols.score(ols_df, house.by_train))

#Ensemble
ensemble = -0.522 + stacked_pred*(0.226) + xgb_pred*(-0.307) + model_rf_pred*1.124
ensemble

#submission
sub = pd.DataFrame()
sub['Id'] = house.bx_test_ids
sub['SalePrice'] = np.expm1(lasso_prediction)
sub.to_csv('submission.csv',index=False)
