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
    "RNN_regressor": RadiusNeighborsRegressor(), # Predicts NaN's :S
        }

mdf = pd.read_csv("data/sales_train.csv")

print(mdf.head())
