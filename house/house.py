import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import missingno as msno
import xgboost as xgb

from statsmodels.formula.api import ols
from statsmodels.genmod import generalized_linear_model

from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


# A class to hold our housing data
class House():
    def __init__(self, train_data_file, test_data_file):
        train = pd.read_csv(train_data_file)
        test = pd.read_csv(test_data_file)
        self.all = pd.concat([train,test], ignore_index=True, sort=True)
        self.all['test'] = self.all.SalePrice.isnull()

    def train(self):
        return(self.all[~self.all['test']])

    def test(self):
        return(self.all[self.all['test']])

    def log_transform(self, variable):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.distplot(variable, bins=50)
        plt.title('Original')
        plt.subplot(1,2,2)
        sns.distplot(np.log1p(variable), bins=50)
        plt.title('Log transformed')
        plt.tight_layout()

    def corr_matrix(self, data, column_estimate, k=10, cols_pair=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']):
        corr_matrix = data.corr()
        sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='coolwarm')
        plt.figure()

        self.price_corr_cols = corr_matrix.nlargest(k, column_estimate)[column_estimate].index
        cm = np.corrcoef(data[self.price_corr_cols].values.T)
        sns.set(font_scale=1.25)
        f, ax = plt.subplots(figsize=(12, 9))
        hm = sns.heatmap(cm, cbar=True, cmap='coolwarm', annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        plt.figure()

        sns.set()
        sns.pairplot(data[cols_pair], size = 2.5)
        plt.show()

    def missing_stats(self):
        # Basic Stats
        self.all.info()

        # Heatmap
        sns.heatmap(self.all.isnull(), cbar=False)
        col_missing=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        col_missing.remove('SalePrice')
        print(col_missing)
        msno.heatmap(self.all)
        plt.figure()
        msno.heatmap(self.all[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']])
        plt.figure()
        msno.heatmap(self.all[['GarageCond', 'GarageFinish', 'GarageFinish', 'GarageQual','GarageType', 'GarageYrBlt']])
        plt.figure()
        msno.dendrogram(self.all)
        plt.figure()

        # Bar chart
        if len(col_missing) != 0:
            plt.figure(figsize=(12,6))
            np.sum(self.all[col_missing].isnull()).plot.bar(color='b')

            # Table
            print(pd.DataFrame(np.sum(self.all[col_missing].isnull())))
            print(np.sum(self.all[col_missing].isnull())*100/self.all[col_missing].shape[0])

    def remove_outliers(self):
        self.all = self.all.drop(self.all[(self.all['GrLivArea']>4000) & (self.all['SalePrice']<300000)].index)
        #self.all = self.all.drop(self.all[(self.all['LotFrontage']>150)].index)

    def distribution_charts(self):
        for column in self.all.columns:
            print( "Graphing " + column)
            if self.all[column].dtype in ['object', 'int64']:
                plt.figure()
                self.all.groupby([column,'test']).size().unstack().plot.bar()

            elif self.all[column].dtype in ['float64']:
                plt.figure(figsize=(10,5))
                sns.distplot(self.all[column][self.all[column]>0])
                plt.title(column)


    def sale_price_charts(self):
        for i, column in enumerate(self.all.columns):
            plt.figure(i)
            if column == 'SalePrice':
                pass
            elif self.all[column].dtype == 'float64':
                data = pd.concat([self.all['SalePrice'], self.all[column]], axis=1)
                data.plot.scatter(x=column, y='SalePrice', ylim=(0,800000))
            else:
                var = column
                data = pd.concat([self.all['SalePrice'], self.all[var]], axis=1)
                f, ax = plt.subplots(figsize=(16, 8))
                fig = sns.boxplot(x=var, y="SalePrice", data=data)
                fig.axis(ymin=0, ymax=800000)


    def clean(self,house_config):
        columns_with_missing_data=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        columns_with_missing_data.remove('SalePrice')

        self.bx_test_ids = self.all[self.all['test']].Id
        self.all.drop('Id', axis=1, inplace=True)

        for column in columns_with_missing_data:
            impute_method = house_config[column]['imputation_method']
            col = self.all[column]
            #print( 'Cleaning ' + str(np.sum(self.all[column].isnull())) + ' data entries for column: ' + column + ' with method ' + impute_method)

            if impute_method  == "mean()":
                col.fillna(self.all[column].mean(),inplace=True)
            if impute_method == "mode()":
                col.fillna(self.all[column].mode()[0],inplace=True)
            elif column == "LotFrontage":
                self.all["LotFrontage"] = self.all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
            elif len(impute_method) != 0:
                self.all[column] = col.fillna(impute_method)
            elif self.all[column].dtype == 'object':
                self.all[column] = col.fillna('None')
            else:

                print( 'Uh oh!!! No cleaning strategy for:' + column )


    # Takes a house config as input and converts types if the length of the dtype is not zero.
    def convert_types(self, house_config):
        for house_variable_name, house_variable_value in house_config.items():
            if len(house_variable_value['dtype']) != 0:
                #print("assigning " + house_variable_name + " as type " + house_variable_value['dtype'])
                self.all[house_variable_name] = self.all[house_variable_name].astype(house_variable_value['dtype'])

    def add_features(self):
        self.all['TotalSF'] = self.all['1stFlrSF'] + self.all['2ndFlrSF']
        self.all['LastConstructionYear'] = list(map(max, self.all['YearBuilt'].values, self.all['YearRemodAdd'].values))
        self.drop_columns = []

    def ordinal_features(self, house_config):
        self.categorical_columns = [x for x in self.all.columns if self.all[x].dtype == 'object' ]
        self.non_categorical_columns = [x for x in self.all.columns if self.all[x].dtype != 'object' ]

        for column in self.categorical_columns:
            for member_name, member_dict in house_config[column]['members'].items():
                if member_dict['ordinal'] != 0:
                    #print( "Replacing " + member_name + " with " + str(member_dict['ordinal']) + " in column " + column)
                    self.all[column].replace(member_name, member_dict['ordinal'], inplace=True)

            #print( "Column " + column + " now has these unique values " + ' '.join(self.all[column].unique()))

    def one_hot_features(self):
        self.use_columns = self.non_categorical_columns + self.categorical_columns
        self.encoded_all = pd.get_dummies(self.all[self.use_columns], drop_first=True, dummy_na=True)
        self.save_test_train_data()


    def label_encode(self, cols=[]):
        # TBD: Take the cols as input if we want to mix label and one hot encoding.
        self.encoded_all = self.all.copy()
        for c in self.encoded_all.columns:
            if self.encoded_all[c].dtype == 'object':
                lce = LabelCountEncoder()
                self.encoded_all[c] = lce.fit_transform(self.encoded_all[c])
        self.save_test_train_data()


    def save_test_train_data(self):
        #print(self.encoded_all.head())
        drop_columns = self.drop_columns + ['test','SalePrice']
        self.bx_train = self.encoded_all[~self.encoded_all['test']].drop(drop_columns, axis=1)
        self.by_train = np.log1p(self.encoded_all[~self.encoded_all['test']].SalePrice)

        self.bx_test = self.encoded_all[self.encoded_all['test']].drop(drop_columns, axis=1)


    def box_cox(self):
        #Refresh the index of the numerical features
        numeric_feats = self.all.dtypes[self.all.dtypes == "float64"].index

        #Calculate skewness
        skewed_feats = self.all[numeric_feats].drop(['SalePrice'], axis=1).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        #print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        #print(skewness.head(10))

        #exctract the features with skewness higher than 75%
        skewness = skewness[abs(skewness.Skew)>0.75]
        #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            #print(feat)
            self.all[feat] = boxcox1p(self.all[feat], lam)


    def sale_price_charts(self):
        for i, column in enumerate(self.all.columns):
            plt.figure(i)
            if column == 'SalePrice':
                pass
            elif self.all[column].dtype == 'float64':
                data = pd.concat([self.all['SalePrice'], self.all[column]], axis=1)
                data.plot.scatter(x=column, y='SalePrice', ylim=(0,800000))
            else:
                var = column
                data = pd.concat([self.all['SalePrice'], self.all[var]], axis=1)
                f, ax = plt.subplots(figsize=(16, 8))
                fig = sns.boxplot(x=var, y="SalePrice", data=data)
                fig.axis(ymin=0, ymax=800000)


    def statsmodel_linear_regression(self,y=['SalePrice'], X=['GrLivArea']):
        x = sm.add_constant(self.train()[X])
        y = self.train()[y]
        model = sm.OLS(y,x)
        results = model.fit()
        print(results.summary())


    def fit_and_predict(self, model, name=""):
        model.fit(self.bx_train.values, self.by_train)
        self.prediction = model.predict(self.bx_test)
        self.save_last_prediction()
        return(self.prediction)


    def save_last_prediction(self):
        submission = pd.DataFrame()
        submission['Id'] = self.bx_test_ids
        submission['SalePrice'] = np.expm1(self.prediction)
        file_name = 'submission.csv'
        submission.to_csv('submission.csv',index=False)


    def rmsle_cv(self,model,n_folds=5):
        kf = KFold(n_folds, shuffle=True).get_n_splits(self.bx_train.values)
        rmse = np.sqrt(-cross_val_score(model, self.bx_train.values, self.by_train.values, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)



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
