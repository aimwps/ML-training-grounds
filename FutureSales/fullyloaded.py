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
tree_regressors = {
    "Decision_tree_regressor": DecisionTreeRegressor(),
    "AdaBoost_regressor": AdaBoostRegressor(),
    "Extra_trees_regressor": ExtraTreesRegressor(),
    "Random_forest_regressor": RandomForestRegressor(), # Takes 55 seconds
    "GBM_regressor": GradientBoostingRegressor(), #Takes forever
    "HGB_regressor": HistGradientBoostingRegressor(),
    "CATBoost_regressor": CatBoostRegressor(verbose=0),
    "lightgbm_regressor": LGBMRegressor(),
        }
mult_regeressors = {
    "Linear_regression": LinearRegression(), ### Dont use results were awful
    "Ridge_regressor": Ridge(),
    "SVM_regressor": SVR(), # Takes 150  seconds
    "MLP_regressor": MLPRegressor(),
    "SGD_regressor": SGDRegressor(),
    "KNN_regressor": KNeighborsRegressor(),
    "BR_regressor" : BayesianRidge(),
    #"RNN_regressor": RadiusNeighborsRegressor(), # Predicts NaN's :S
        }

all_models = {**tree_regressors, **mult_regeressors}
################################################################################
## Construct the dataframe, merge other other dataframes into it.
mdf = pd.read_csv("data/sales_train.csv")
items = pd.read_csv("data/en_items.csv")
shops = pd.read_csv("data/en_shops.csv")
categories = pd.read_csv("data/en_categories.csv")
test = pd.read_csv("data/test.csv")
mdf.sort_values(["date", "date_block_num", "shop_id", "item_id"], inplace=True)
mdf = mdf.merge(items, on="item_id", how="left")
mdf = mdf.merge(categories, left_on="item_category_id", right_on="category_id", how="left")
mdf = mdf.merge(shops, on="shop_id", how ="left")
mdf = mdf.drop('item_category_id', axis=1)
#print(mdf['item_cnt_day'].describe())
mdf['item_cnt_day'] = mdf['item_cnt_day'].clip(0,20)
#print(mdf['item_cnt_day'].describe())

def get_shop_data(df, shopid): #'shop_id'
    return df[df["shop_id"] == shopid]

def get_month_data(df, monthid):
    return df[df["date_block_num"]==monthid]

def group_month_sales(df):
    target_values = df.groupby('item_id')['item_cnt_day'].agg('sum')
    x = target_values.describe()
    print(x)


### From the test data extract
## The Shop name
## The Item name
#### Perform feature engineering on the Shop name
#### Perform feature engineering on the item name



### For the training data
### For grouping the month
mdf33 = get_month_data(mdf, 33)

gbmdf33 = mdf33.groupby(['shop_id', "item_id"])['item_cnt_day'].sum().reset_index()
X = gbmdf33[['shop_id', "item_id"]]
Y = gbmdf33['item_cnt_day'].clip(0,20)

# TX = test[['shop_id', "item_id"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
#
# GBR.fit(X_train, Y_train)
# preds = GBR.predict(X_test)
# score = metrics.mean_squared_error(Y_test, preds, squared=False)
# print(score)

# results = pd.DataFrame({"Model_name":[], "RSME": [], "RUNTIME": []})
# for model_name, model in all_models.items():
#     print(f"Working on {model_name}")
#     start_time = time.time()
#     model.fit(X_train, Y_train)
#     pred = model.predict(X_test)
#     total_time = time.time() - start_time
#     score = metrics.mean_squared_error(Y_test, pred, squared=False)
#     results = results.append({
#                     "Model_name":model_name,
#                     "RSME": score,
#                     "RUNTIME": total_time}
#                     ,ignore_index=True)
# results_ord = results.sort_values(by=['RSME'], ascending=False, ignore_index=True)
# print(results_ord.head(20))


def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = RandomForestRegressor(max_depth=rf_max_depth)


    X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)
print(" Value: ", study.best_trial.value)
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")



# na = gbmdf33.isna().sum()
# print(gbmdf33.sample(100))
#


# shop = get_shop_data(mdf,25)
# print(shop.sample(50))
# print(shop.shape)
#
# shop_month = get_month_data(shop, 33)
# print(shop_month.sample(50))
# print(shop_month.shape)
#
# item_month_sales = group_month_sales(shop_month)
# print(item_month_sales.head())
# print(item_month_sales.shape)
# print(item_month_sales.describe())
