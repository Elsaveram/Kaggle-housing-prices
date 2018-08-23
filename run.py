
## Data loading
# %%
%reload_ext autoreload
%autoreload 2
import math
import json
from house import *
from config import *
del house
house = House('data/train.csv','data/test.csv')
# %%

#%%
# Clean/impute data
#house.cleanRP()
house.clean()
# Convert types after clenaing and filling in all NAs
house.convert_types(HOUSE_CONFIG)
##Feature Engeneneering:
house.engineer_features(HOUSE_CONFIG)
house.xgboost
house.sk_random_forest(1000)
#%%

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
