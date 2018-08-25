import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from  statsmodels.genmod import generalized_linear_model
from scipy.stats import skew
from scipy.special import boxcox1p

import missingno as msno

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from LabelClassEncoder import *

# A class to hold our housing data
class House():
    def __init__(self, train_data_file, test_data_file):
        train = pd.read_csv(train_data_file)
        test = pd.read_csv(test_data_file)
        self.all = pd.concat([train,test], ignore_index=True)
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

        cols = corr_matrix.nlargest(k, column_estimate)[column_estimate].index
        cm = np.corrcoef(data[cols].values.T)
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


    def relation_stats(self, x, y, z):
        # x vs y scatter
        plt.figure()
        self.all.plot.scatter(x, y)
        print(self.all[[x, y]].corr(method='pearson'))

        # z vs x box
        df_config = self.all[[z, x]]
        df_config.boxplot(by=z, column=x)
        mod_2 = ols( x + ' ~ ' + z, data=df_config).fit()

        aov_table = sm.stats.anova_lm(mod_2, typ=2)
        print(aov_table)

        #LotFrontage vs LotShape #significant
        df_frontage = self.all[['LotShape', 'LotFrontage']]
        df_frontage.boxplot(by='LotShape', column='LotFrontage')

        mod = ols('LotFrontage ~ LotShape', data=df_frontage).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        print(aov_table)


    def clean(self):
        columns_with_missing_data=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        columns_with_missing_data.remove('SalePrice')

        self.bx_test_ids = self.all[self.all['test']].Id
        self.all.drop('Id', axis=1, inplace=True)

        for column in columns_with_missing_data:
            col_data = self.all[column]
            #print( 'Cleaning ' + str(np.sum(col_data.isnull())) + ' data entries for column: ' + column )

            if column in ['Electrical']:
                self.all[column] = [ self.all[column].mode()[0] if pd.isnull(x) else x for x in self.all[column]]
            elif  column=='GarageYrBlt':
                self.all[column] = [0 if pd.isnull(x) else x for x in self.all['GarageYrBlt']]
                self.all[column] = [0 if x == 'NA' else x for x in self.all['GarageYrBlt']]
            elif column == 'LotFrontage':
                self.all["LotFrontage"] = self.all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
            elif column == 'GarageYrBlt':
                # TBD: One house has a detached garage that could be caclulatd based on the year of construction.
                self.all[column] = [ 'NA' if pd.isnull(x) else x for x in self.all[column]]
            elif column in ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','MasVnrArea']:
                self.all[column] = [ 0 if pd.isnull(x) else x for x in self.all[column]]
            elif column == 'Functional':
                self.all[column] = [ 'Typ' if pd.isnull(x) else x for x in self.all[column]]
            elif col_data.dtype == 'object':
                self.all[column] = [ "None" if pd.isnull(x) else x for x in self.all[column]]
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
        self.drop_columns = []


    def ordinal_features(self, house_config):
        self.categorical_columns = [x for x in self.all.columns if self.all[x].dtype == 'object' ]
        self.non_categorical_columns = [x for x in self.all.columns if self.all[x].dtype != 'object' ]

        # TBD: do something with ordinals!!!!!
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
        self.by_train = self.encoded_all[~self.encoded_all['test']].SalePrice

        self.bx_test = self.encoded_all[self.encoded_all['test']].drop(drop_columns, axis=1)

    def box_cox(self):
        #Refresh the index of the numerical features
        numeric_feats = self.all.dtypes[self.all.dtypes == "float64"].index

        #Calculate skewness
        skewed_feats = self.all[numeric_feats].drop(['SalePrice'], axis=1).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        print(skewness.head(10))

        #exctract the features with skewness higher than 75%
        skewness = skewness[abs(skewness) > 0.75]
        print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            print(feat)
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

    def rmse_cv(self,model, x, y, k=5):
        rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
        return(np.mean(rmse))


    def statsmodel_linear_regression(self,y=['SalePrice'], X=['GrLivArea']):
        x = sm.add_constant(self.train()[X])
        y = self.train()[y]
        model = sm.OLS(y,x)
        results = model.fit()
        print(results.summary())


    def test_train_split(self, dataset):
        x=dataset[~dataset['test']].drop(['test','SalePrice'], axis=1).astype(object)
        y=dataset[~dataset['test']].SalePrice
        try:
            self.x_train
        except:
            print('DOING SPLITS!!!!')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y)


    def sk_random_forest(self, dataset, num_est=500):
        self.test_train_split(dataset)

        model_rf = RandomForestRegressor(n_estimators=num_est, n_jobs=-1)
        model_rf.fit(self.x_train, self.y_train)
        self.rf_pred = model_rf.predict(self.x_test)

        #plt.figure(figsize=(10, 5))
        #plt.scatter(self.y_test, self.rf_pred, s=20)
        #plt.title('Predicted vs. Actual')
        #plt.xlabel('Actual Sale Price')
        #plt.ylabel('Predicted Sale Price')

        #plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)])
        #plt.tight_layout()

        print(self.rmse_cv(model_rf, self.x_train, self.y_train))
