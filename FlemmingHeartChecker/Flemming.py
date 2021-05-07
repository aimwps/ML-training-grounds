import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, gc
from sklearn                    import compose
from sklearn                    import impute
from sklearn                    import metrics
from sklearn                    import model_selection
from sklearn.linear_model       import LogisticRegression, LinearRegression, RidgeClassifier, Lasso
from sklearn.neural_network     import MLPClassifier
from sklearn                    import svm
from sklearn.preprocessing      import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.cluster            import KMeans
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.naive_bayes        import GaussianNB, MultinomialNB
from sklearn.model_selection    import cross_val_score, ShuffleSplit, GridSearchCV, train_test_split, StratifiedKFold, cross_val_predict
from sklearn                    import pipeline
from sklearn.tree               import DecisionTreeClassifier
from sklearn.experimental       import enable_hist_gradient_boosting # for HistGradientBoostingClassifier
from sklearn.ensemble           import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost                    import XGBClassifier
from lightgbm                   import LGBMClassifier
from catboost                   import CatBoostClassifier
import streamlit as st

##########################################################################################################
######## POTENTIAL CLASSIFIERS
mult_classifiers = {
        "LM Logistic Regression": LogisticRegression(),
        "LM Ridge": RidgeClassifier(),
        "NN Multi layer Perceptron": MLPClassifier(random_state=909, verbose=0),
        "SVM Linear": svm.SVC(kernel='linear'),
        "SVM RBF": svm.SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(),
        "BM Guassian Naive Bayes": GaussianNB(),

}

tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Trees":ExtraTreesClassifier(),
        "Random Forest":RandomForestClassifier(),
        "AdaBoost":AdaBoostClassifier(),
        "Skl GBM":GradientBoostingClassifier(),
        "Skl HistGBM":HistGradientBoostingClassifier(),
        "XGBoost":XGBClassifier(use_label_encoder=False, verbose=0),
        "LightGBM":LGBMClassifier(),
        "CatBoost":CatBoostClassifier(verbose=0)
}

##########################################################################################################
######## DATASET FEATURE INFORMATIION

cat_vars =  [
        "sex",# gender 1/0
        "cp", # chest pain type - 1=angina, 2=atypical angina, 3=non-angina, 4=asympotmatic
        "fbs",# 1/0 true fasting blood pressue >120mg
        "restecg", # 0=normal, 1=ST-T(?!), 2=hypertrophy
        "exang", # 1/0 true false exercise induced angina
        "slope", # 1= upsloping, 2=flat, 3=downsloping
        "thal", # 3=normal, 6=defect, 7=reversable defect
        "ca", # number of  major  vessels coloured by flousopy ?! 0-3
        ]

num_vars = [
        "age", # age value in years
        "trestbps", # resting blood pressue
        "chol", # cholestral
        "thalach", # maximum heart rate achieved
        "oldpeak" # ST(?!) 3depression induced by exercise relative to rest
        ]


##########################################################################################################
######## multiplicative PIPELINES
## For Categorical multiplicative models
cat_mult_pipeline = pipeline.Pipeline(steps=[
    ('One Hot', OneHotEncoder(handle_unknown="ignore")),
    ])

## For Numerical multipicative models
num_mult_pipeline = pipeline.Pipeline(steps=[
    ("Quantile transformer", QuantileTransformer(n_quantiles=100,output_distribution='normal')),
    ])

## For bringing nultiplicative numerical / Categorical together
mult_prepro = compose.ColumnTransformer(transformers=[
    ('multnum', num_mult_pipeline, num_vars),
    ('multcat', cat_mult_pipeline, cat_vars),
    ], remainder='passthrough')

##########################################################################################################
######## multiplicative PIPELINES
## For numerical tree models
# num_tree_pipeline = pipeline.Pipeline(steps=[
#     # nothing and not needed??
#     ])

## for categorical tree models
cat_tree_pipeline =pipeline.Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ])

#For bringing tree numerical / Categorical together
tree_prepro = compose.ColumnTransformer(transformers=[
    ('treecat', cat_tree_pipeline, cat_vars),
    ], remainder='passthrough')


##########################################################################################################
######## Create the piples for models
tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
mult_classifiers = {name: pipeline.make_pipeline(mult_prepro, model) for name, model in mult_classifiers.items()}

## Combine the pipelines
all_pipelines = {**tree_classifiers, **mult_classifiers}


##########################################################################################################
######## Create the dataframe and split data
mheart = pd.read_csv('heart.csv')
XX = mheart.drop('target', axis=1)
YY = mheart['target']

print(f"X SHAPE {XX.shape}, Y SHAPE {YY.shape}")

xx_train, xx_val, yy_train, yy_val = train_test_split(XX, YY)

##########################################################################################################
######## For running a all models and getting results
results = pd.DataFrame({'Model': [], "Accuracy": [], "Bal Acc.": [], 'Time': []})
skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for model_name, model in all_pipelines.items():
    start_time = time.time()

## no cross validation
    # model.fit(xx_train, yy_train)
    # pred = model.predict(xx_val)
## with cross validation
    pred = model_selection.cross_val_predict(model, XX, YY, cv=skf)

    total_time = time.time() - start_time
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(YY, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(YY, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
