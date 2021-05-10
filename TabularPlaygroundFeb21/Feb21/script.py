import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import time
from sklearn                import pipeline      # Pipeline
from sklearn                import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn                import impute
from sklearn                import compose
from sklearn                import model_selection # train_test_split
from sklearn                import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn                import set_config
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm            import SVR
from sklearn.linear_model   import Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors      import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.experimental   import enable_hist_gradient_boosting
from sklearn.ensemble       import HistGradientBoostingRegressor
from catboost               import CatBoostRegressor
from lightgbm               import LGBMRegressor

tree_classifiers = {
    "Decision_tree_regressor": DecisionTreeRegressor(),
    "AdaBoost_regressor": AdaBoostRegressor(),
    "Extra_trees_regressor": ExtraTreesRegressor(),
    #"Random_forest_regressor": RandomForestRegressor(), # Takes 55 seconds
    #"GBM_regressor": GradientBoostingRegressor(), Takes forever
    "HGB_regressor": HistGradientBoostingRegressor(),
    "CATBoost_regressor": CatBoostRegressor(verbose=0),
    "lightgbm_regressor": LGBMRegressor(),
        }
mult_classifiers = {
    #"Linear_regression": LinearRegression(), ### Dont use results were awful
    "Ridge_regressor": Ridge(),
    #"SVM_regressor": SVR(), # Takes 150  seconds
    "MLP_regressor": MLPRegressor(),
    "SGD_regressor": SGDRegressor(),
    "KNN_regressor": KNeighborsRegressor(),
    "BR_regressor" : BayesianRidge(),
    #"RNN_regressor": RadiusNeighborsRegressor(), # Predicts NaN's :S
        }

path = "./tabular-playground-feb21/"
df = pd.read_csv(path+"train.csv", index_col='id')
df_t = pd.read_csv(path+"test.csv", index_col='id')
sub = pd.read_csv(path+"sample_submission.csv", index_col='id')
df = df.sample(10000)
x = df.drop(['target'], axis=1)
y = df['target']
cat_vars = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
num_vars = ['cont0', 'cont1', 'cont2', 'cont3',  'cont4',  'cont5',  'cont6',
            'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13']

mult_num_var = pipeline.Pipeline(steps=[
    ('STANDARDSCALER',  preprocessing.StandardScaler()),
    ])

mult_cat_var = pipeline.Pipeline(steps=[
    ('ONEHOTENCODER', preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])

mult_prepro = compose.ColumnTransformer(transformers=[
    ("MULT_NUM_VAR", mult_num_var, num_vars),
    ("MULT_CAT_VAR", mult_cat_var, cat_vars),
    ], remainder='passthrough')


tree_cat_var = pipeline.Pipeline(steps=[
    ('ONEHOTENCODER', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

tree_prepro = compose.ColumnTransformer(transformers=[
    #("TREE_NUM_VAR", tree_num_var, num_vars),
    ("TREE_CAT_VAR", tree_cat_var, cat_vars),
    ], remainder='passthrough')

tree_pipes = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
mult_pipes = {name: pipeline.make_pipeline(mult_prepro, model) for name, model in mult_classifiers.items()}
all_pipelines = {**tree_pipes,**mult_pipes}

CBR_params = {
    "abc__learning_rate": [0.1, 0.01, 0.001, 0.0001]
}


pl = pipeline.make_pipeline(tree_prepro, CatBoostRegressor())
grid_search = model_selection.GridSearchCV(pl, param_grid=CBR_params)
grid_score = grid_search.fit(x,y)

print(grid_score.best_params_)
