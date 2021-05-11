import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, gc
from sklearn                    import compose, impute, metrics, model_selection, pipeline
from sklearn.preprocessing      import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.model_selection    import cross_val_score, ShuffleSplit, GridSearchCV, train_test_split, StratifiedKFold, cross_val_predict
from sklearn.tree               import DecisionTreeRegressor
from sklearn.ensemble           import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network     import MLPRegressor
from sklearn.svm                import SVR
from sklearn.linear_model       import Ridge, Lasso, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.neighbors          import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.experimental       import enable_hist_gradient_boosting
from sklearn.ensemble           import HistGradientBoostingRegressor
from catboost                   import CatBoostRegressor
from lightgbm                   import LGBMRegressor
from itertools                  import product
import optuna


################################################################################
## Construct the dataframe
mdf = pd.read_csv("data/sales_train.csv")
items = pd.read_csv("data/en_items.csv")
shops = pd.read_csv("data/en_shops.csv")
categories = pd.read_csv("data/en_categories.csv")
test = pd.read_csv("data/test.csv")


################################################################################
## Merge corresponding datafrom other data frames
mdf.sort_values(["date", "date_block_num", "shop_id", "item_id"], inplace=True)
mdf = mdf.merge(items, on="item_id", how="left")
mdf = mdf.merge(categories, left_on="item_category_id", right_on="category_id", how="left")
mdf = mdf.merge(shops, on="shop_id", how ="left")
mdf = mdf.drop('item_category_id', axis=1)
mdf['item_cnt_day'] = mdf['item_cnt_day'].clip(0,20)

################################################################################
## Get total sales from previous month
mdf33 = mdf[mdf['date_block_num']==33]
gbmdf33 = mdf33.groupby(['shop_id', "item_id"])['item_cnt_day'].sum().reset_index()
test = test.merge(gbmdf33, on=['shop_id', "item_id"], how="left")
test = test.fillna(0)
test['item_cnt_month'] = test['item_cnt_day'].clip(0,20)

################################################################################
## FOR GENERATING MONTHLY ITEM SALES ACROSS ALL STORES

#### Used for generating data for the training set
def generate_total_item_sales_all(df):
    for month_num in sorted(df['date_block_num'].unique()):
        month_df = df[df['date_block_num']==month_num]#
        group_product_month = month_df.groupby('item_id')['item_cnt_day'].sum().reset_index()
        df = df.merge(group_product_month, on=["item_id"], how="left")
        df.rename(columns={"item_cnt_day_x":"item_cnt_day", "item_cnt_day_y": f"item_total_month_{month_num}"}, inplace=True)
        df[f"item_total_month_{month_num}"].fillna(0, inplace=True)

        print(df.head())
        print(df.info())
    df.to_csv("data/prodsales.csv")
    return df

#### Used for generating data for a prediction
def generate_total_item_sales_by_ID(train_df, i_id):
    results = {"item_id": [i_id],}
    for month_num in sorted(train_df['date_block_num'].unique()):
        month_df = train_df[train_df['date_block_num']== month_num]
        print(month_df.head())
        a = month_df[month_df['item_id'] == i_id]['item_cnt_day'].sum()

        results[f"item_total_month_{month_num}"] = [month_df[month_df['item_id'] == i_id]['item_cnt_day'].sum()]
        print(results)

    for k,v in results.items():
        print(f"label: {k}, result: {v}")

    return pd.DataFrame(results)

################################################################################
## FOR GENERATING MONTHLY SHOP SALES ACROSS ALL STORES

#### Used for generating data for the training set
def generate_total_shop_sales_all(df):
    for month_num in sorted(df['date_block_num'].unique()):
        month_df = df[df['date_block_num']==month_num]#
        group_shop_month = month_df.groupby('shop_id')['item_cnt_day'].sum().reset_index()
        df = df.merge(group_shop_month, on=["shop_id"], how="left")
        print(df.info())
        df.rename(columns={"item_cnt_day_x":"item_cnt_day", "item_cnt_day_y": f"shop_total_month_{month_num}"}, inplace=True)
        df[f"shop_total_month_{month_num}"].fillna(0, inplace=True)
        print(df.sample(10))
    print(df.head())
    print(df.tail())
    print(df.info())
    return df

#### Used for generating data for a prediction
def generate_total_shop_sales_by_ID(train_df, s_id):
    results = {"shop_id": [s_id],}
    for month_num in sorted(train_df['date_block_num'].unique()):
        month_df = train_df[train_df['date_block_num']== month_num]
        results[f"shop_total_month_{month_num}"] = [month_df[month_df['shop_id'] == s_id]['item_cnt_day'].sum()]
        print(results)

    for k,v in results.items():
        print(f"label: {k}, result: {v}")

    return pd.DataFrame(results)

################################################################################
## FOR REORDERING DATA FRAME FOR MODEL SUBMISSION:

def reorder_dataframe(df):
    shop_sales = [f"shop_total_month_{i}" for i in sorted(df['date_block_num'].unique)]
    item_sales = [f"item_total_month_{i}" for i in sorted(df['date_block_num'].unique)]
    df = df[['date', 'date_block_num', "shop_id", "item_id","item_price", "item_cnt_day", "item_name", "category_name", "category_id", "shop_name"]+ shop_sales + item_sales]

    print(df.info())
    return df

print(mdf.shape)
print(mdf.info())
mdf = generate_total_shop_sales_all(mdf)
print(mdf.shape)
print(mdf.info())
mdf = generate_total_item_sales_all(mdf)
print(mdf.shape)
print(mdf.info())
mdf = reorder_dataframe(mdf)
print(mdf.shape)
print(mdf.info())


##  Each product has a total sales for each month
##  When predicting shop/product combo
##  Look up for that product the total sales for each month
##  generate total_product_1, total_product_2, total_product_3 etc
