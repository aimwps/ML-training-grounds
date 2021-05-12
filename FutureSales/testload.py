import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, gc, optuna
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
from geopy.geocoders            import Nominatim
from functools                  import partial
from geopy.extra.rate_limiter   import RateLimiter


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
## FOR GENERATING MONTHLY SHOP SALES ACROSS ALL ITEMS
def generate_total_shop_sales_all(df):
    for month_num in sorted(df['date_block_num'].unique()):
        month_df = df[df['date_block_num']==month_num]#
        group_shop_month = month_df.groupby('shop_id')['item_cnt_day'].sum().reset_index()
        df = df.merge(group_shop_month, on=["shop_id"], how="left")
        df.rename(columns={"item_cnt_day_x":"item_cnt_day", "item_cnt_day_y": f"shop_total_month_{month_num}"}, inplace=True)
        df[f"shop_total_month_{month_num}"].fillna(0, inplace=True)
    return df

#### Used for generating data for a prediction
def generate_total_shop_sales_by_ID(train_df, s_id):
    results = {"shop_id": [s_id],}
    for month_num in sorted(train_df['date_block_num'].unique()):
        month_df = train_df[train_df['date_block_num']== month_num]
        results[f"shop_total_month_{month_num}"] = [month_df[month_df['shop_id'] == s_id]['item_cnt_day'].sum()]
    return pd.DataFrame(results)


###############################################################################
## FOR GENERATING MONTHLY ITEM SALES ACROSS ALL STORES
def generate_total_item_sales_all(df):
    for month_num in sorted(df['date_block_num'].unique()):
        month_df = df[df['date_block_num']==month_num]#
        group_product_month = month_df.groupby('item_id')['item_cnt_day'].sum().reset_index()
        df = df.merge(group_product_month, on=["item_id"], how="left")
        df.rename(columns={"item_cnt_day_x":"item_cnt_day", "item_cnt_day_y": f"item_total_month_{month_num}"}, inplace=True)
        df[f"item_total_month_{month_num}"].fillna(0, inplace=True)
    return df

#### Used for generating data for a prediction
def generate_total_item_sales_by_ID(train_df, i_id):
    results = {"item_id": [i_id],}
    for month_num in sorted(train_df['date_block_num'].unique()):
        month_df = train_df[train_df['date_block_num']== month_num]
        results[f"item_total_month_{month_num}"] = [month_df[month_df['item_id'] == i_id]['item_cnt_day'].sum()]
    return pd.DataFrame(results)


###############################################################################
## FOR GENERATING MONTHLY ITEM SALES ACROSS ALL STORES
def generate_total_item_sales_all(df):
    for month_num in sorted(df['date_block_num'].unique()):
        month_df = df[df['date_block_num']==month_num]#
        group_product_month = month_df.groupby('item_id')['item_cnt_day'].sum().reset_index()
        df = df.merge(group_product_month, on=["item_id"], how="left")
        df.rename(columns={"item_cnt_day_x":"item_cnt_day", "item_cnt_day_y": f"item_total_month_{month_num}"}, inplace=True)
        df[f"item_total_month_{month_num}"].fillna(0, inplace=True)
    return df

#### Used for generating data for a prediction
def generate_total_item_sales_by_ID(train_df, i_id):
    results = {"item_id": [i_id],}
    for month_num in sorted(train_df['date_block_num'].unique()):
        month_df = train_df[train_df['date_block_num']== month_num]
        results[f"item_total_month_{month_num}"] = [month_df[month_df['item_id'] == i_id]['item_cnt_day'].sum()]
    return pd.DataFrame(results)

################################################################################
## PREPROCESS CATEGORICAL CSV
def get_multi_categories(df):
    df[['category_1', 'category_2']] = df['category_name'].str.split("-", n=1, expand=True)
    df['category_2'] = df['category_2'].str.replace("\([^)]*\)","", regex=True)
    return df


################################################################################
## PREPROCESS SHOP DATA
def get_store_location(df):
    geolocator = Nominatim(user_agent="future_sales")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    df['shop_name'] = df["shop_name"].str.strip("!").str.strip().str.replace(". ", ".", n=1, regex=False)
    df['city'] = df['shop_name'].str.split(" ", n=1, expand=True)[0]
    df['city'] = df['city'].str.strip()
    df['city'] = df['city'].str.replace("RostovNaDonu","Rostov-on-Don", regex=False)
    df['city'] = df['city'].str.replace("RostovNaDon","Rostov-on-Don", regex=False)
    df['location'] = df['city'].apply(partial(geocode, language="en"))
    df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    df['latitude'] = df['point'].apply(lambda loc: loc[0])
    df['longitude'] = df['point'].apply(lambda loc: loc[1])
    df.drop(['location', 'point'], axis=1, inplace=True)
    return df

def determine_online_store(df):
    df['online_only'] = df['shop_name'].apply(lambda loc: True if ('Online' in loc or 'Internet' in loc) else False)
    return df

def replace_duplicate_shop_id(df):
    df['AIshop_id'] = df['shop_id'].replace({57:0, 58:1, 11:10, 41:39})
    return df

def transform_shops(df):
    df = get_store_location(df)
    df = replace_duplicate_shop_id(df)
    df = determine_online_store(df)
    return df
#pp_shops = transform_shops(shops)


################################################################################
## PREPROCESS ITEM DATA
















################################################################################
## OLD NOT NEEDED YET




def see_if_missing_item(train_df, test_df):
    uni_train = sorted(list(train_df['item_id'].unique()))
    uni_test = sorted(list(test_df['item_id'].unique()))
    new_entries = []
    accepted_entries = []
    for t in uni_test:
        if t in uni_train:
            accepted_entries.append(t)
        else:
            new_entries.append(t)

    print(len(accepted_entries))
    print(len(new_entries))
    return new_entries




print(see_if_missing_item(items, test ))





























################################################################################
def preprocess_prediction_df(train_df, test_df):
    print(f"SHAPE OF test_DF : {test_df.shape}")
    shop_results = None
    for s_id in test_df['shop_id'].unique():
        if isinstance(shop_results, pd.DataFrame):
            shop_results = pd.concat([shop_results, generate_total_shop_sales_by_ID(train_df, s_id)])
        else:
            shop_results = generate_total_shop_sales_by_ID(train_df, s_id)
    test_df = test_df.merge(shop_results, on="shop_id", how="left")
    print(f"SHAPE OF test_DF : {test_df.shape}")
    item_results = None
    for i, i_id in enumerate(test_df['item_id'].unique()):
        print(f"{i}/{len(test_df['item_id'].unique())}")
        if isinstance(shop_results, pd.DataFrame):
            item_results = pd.concat([item_results, generate_total_item_sales_by_ID(train_df, i_id)])
        else:
            item_results = generate_total_item_sales_by_ID(train_df, s_id)
    test_df = test_df.merge(item_results, on="item_id", how="left")
    print(f"SHAPE OF test_DF : {test_df.shape}")
    return test_df

































################################################################################
## FOR REORDERING DATA FRAME FOR MODEL SUBMISSION:

def reorder_dataframe(df):
    shop_sales = [f"shop_total_month_{i}" for i in sorted(df['date_block_num'].unique())]
    item_sales = [f"item_total_month_{i}" for i in sorted(df['date_block_num'].unique())]
    df = df[['date', 'date_block_num', "shop_id", "item_id","item_price", "item_cnt_day", "item_name", "category_name", "category_id", "shop_name"]+ shop_sales + item_sales]

    print(df.info())
    return df

# print(mdf.shape)
# print(mdf.info())
# mdf = generate_total_shop_sales_all(mdf)
# print(mdf.shape)
# print(mdf.info())
# mdf = generate_total_item_sales_all(mdf)
# print(mdf.shape)
# print(mdf.info())
# mdf = reorder_dataframe(mdf)
# print(mdf.shape)
# print(mdf.info())


##  Each product has a total sales for each month
##  When predicting shop/product combo
##  Look up for that product the total sales for each month
##  generate total_product_1, total_product_2, total_product_3 etc
