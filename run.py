
# %%
%reload_ext autoreload
%autoreload 2
import math
import json
from house import *
from config import *

# %% load data
del house
house = House('data/train.csv','data/test.csv')
# %%


# %% Clean, impute, engineer, and encode data
house.remove_outliers()
house.clean()
house.convert_types(HOUSE_CONFIG)
house.add_features()
house.box_cox()
house.ordinal_features(HOUSE_CONFIG)
#house.label_encode()
house.one_hot_features()
# %%


# %% Create and test models
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

model_rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

lasso = make_pipeline(RobustScaler(), Lasso(alpha= 0.00016102, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha= 0.00016102, l1_ratio=.9, random_state=3))

KRR = KernelRidge()

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,
                            min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
                            subsample=0.5213, silent=1, random_state =7, nthread = -1)

score = house.rmsle_cv(model_rf)
print("model_rf score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

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

rf_pred = house.fit_and_predict(model_rf)
lasso_prediction = house.fit_and_predict(lasso)
enet_prediction = house.fit_and_predict(ENet)
gboost_prediction = house.fit_and_predict(GBoost)
# TBD: Feature name mismatch
#xgb_prediction = house.fit_and_predict(model_xgb)
# %%

from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-5, 2, 30)
grid = GridSearchCV(estimator=Lasso(),param_grid=dict(alpha=alphas), cv=10, scoring='r2')
grid.fit(house.bx_train, house.by_train) # entire datasets were fed here

print(grid.best_params_, grid.best_score_)# score -0.0470788758558
for params, mean_score, scores in grid.grid_scores_:
    print(mean_score, params)



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
ensemble = -0.577 + stacked_pred*(-0.03) + xgb_pred*(-0.251) + model_rf_pred*1.329

#submission
sub = pd.DataFrame()
sub['Id'] = house.bx_test_ids
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
