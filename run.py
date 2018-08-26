
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
##Feature Engeneneering:
house.ordinal_features(HOUSE_CONFIG)
#house.label_encode()
house.one_hot_features()

#%%

house.sk_random_forest(house.encoded_all, 100)

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
    rmse= np.sqrt(-cross_val_score(model, house.bx_train.values, house.by_train.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

model_rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

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

score = rmsle_cv(model_rf)
print("\model_rf score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

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

model_rf.fit(house.bx_train.values, house.by_train)
self.rf_pred = model_rf.predict(house.bx_test)

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

# Averaging models
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

averaged_models = AveragingModels(models = (ENet, GBoost, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], pd.Series(y)[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


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
