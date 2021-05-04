import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as time
from sklearn                    import compose
from sklearn                    import impute
from sklearn                    import metrics
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

##########################################################################################################
### Loading Data
train_data = "train.csv"
kaggle_test = "test.csv"
train_df = pd.read_csv(train_data)
kaggle_test_df = pd.read_csv(kaggle_test)

##########################################################################################################
### Explore data

# print(train_df.describe(), kaggle_test_df.describe())
# ### Check for missing values
# print(train_df.isna().sum()) #= Age: 177, Cabin: 687, Embarked: 2
# print(kaggle_test_df.isna().sum()) #= Age: 86, Cabin: 327, Fare: 1


##########################################################################################################
### preprocess

##### Get a title from the name
def change_name_to_title(df, dropname=True):
    get_passanger_title  = lambda x: x.split(",")[1].split(".")[0].strip()
    df['Title'] = df['Name'].map(get_passanger_title)
    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"
    }
    df['Title'] = df['Title'].map(title_dictionary)
    if dropname:
        df.drop('Name', inplace=True, axis=1)
    return df

##### Fill missing ages
def fill_missing_ages(df, train_df=None):
    if 'Title' not in df.columns:
        df = change_name_to_title(df)
    if train_df:
        group_ages = train_df.groupby(['Title']).agg({"Age": np.mean}).reset_index().set_index('Title')
    else:
        group_ages = df.groupby(['Title']).agg({"Age": np.mean}).reset_index().set_index('Title')
    group_ages['Age'] = group_ages['Age'].round().astype(int)
    transpose_group_ages = group_ages.transpose()
    age_dict = transpose_group_ages.to_dict('list')

    for title, age in age_dict.items():
         df.loc[(df.Age.isna()) & (df.Title == title), "Age"] = age[0]

    return df

pp_train_df = fill_missing_ages(train_df)

print(pp_train_df.describe())

##########################################################################################################
#### Isolate Models, numerical and categorical data
mult_classifiers = {
        #"LM Linear Regression": LinearRegression(), # not useful for classification on titanic
        #"LM Logistic Regression": LogisticRegression(),
        #"LM Ridge": RidgeClassifier(),
        #"LM Lasso": Lasso(), Not useful for titanic dataset
        "NN Multi layer Perceptron": MLPClassifier(random_state=909),
        #"SVM Linear": svm.SVC(kernel='linear'),
        #"SVM RBF": svm.SVC(kernel='rbf'),
        #"KNN": KNeighborsClassifier(),
        #"BM Guassian Naive Bayes": GaussianNB(),
        #"BM Multinominal Naive Bayes":MultinomialNB(),

}

tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Trees":ExtraTreesClassifier(),
        "Random Forest":RandomForestClassifier(),
        "AdaBoost":AdaBoostClassifier(),
        "Skl GBM":GradientBoostingClassifier(),
        "Skl HistGBM":HistGradientBoostingClassifier(),
        "XGBoost":XGBClassifier(use_label_encoder=False),
        "LightGBM":LGBMClassifier(),
        "CatBoost":CatBoostClassifier()
        }
cat_vars  = ['Sex', 'Embarked', 'Title']
num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']

##########################################################################################################
#### Create Pipelines
cat_mult_pipeline = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
    ('One Hot', OneHotEncoder(handle_unknown="error")),
    ])

num_mult_pipeline = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='median')),
    #('Scaler',StandardScaler()),
    ('Quantile Transform',QuantileTransformer(n_quantiles=801)),
    #('Yeo-Johnson', PowerTransformer(method='yeo-johnson')),
    #('Box-Cox', PowerTransformer(method='box-cox')), # all values must be greater than 0
    #('Scaler',StandardScaler),
    ])

mult_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_mult_pipeline, num_vars),
    ('cat', cat_mult_pipeline, cat_vars),
    ], remainder='drop')

#### PIPELINES FOR TREES
num_tree_pipeline = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
    ])

cat_tree_pipeline =pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant',fill_value='missing')),
    ('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_tree_pipeline, num_vars),
    ('cat', cat_tree_pipeline, cat_vars),
    ], remainder='drop')

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
mult_classifiers = {name: pipeline.make_pipeline(mult_prepro, model) for name, model in mult_classifiers.items()}


##########################################################################################################
#### create parameters for gridsearch
param_grid = [
    {
    'mult_prepro__num__imputer__strategy' :  ['mean', 'median'],
    'classifier__solver' :      ['lbfgs', 'sgd', 'adam'],
    'classifier': [MLPClassifier(random_state=909)]
    }
]



##########################################################################################################
#### Run the models

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
x = pp_train_df.drop(columns=["Survived", 'Ticket'])
y = pp_train_df['Survived']
print(x.shape)
print("XXXXXXXXXXXXXXXXXXXXX")


for model_name, model in mult_classifiers.items():
    start_cross_val = time.time()
    #pred = cross_val_predict(model, x, y,cv=skf)
    clf = GridSearchCV(model, param_grid, cv=10, scoring='accuracy').fit(x,y)
    total_cross_val = time.time() - start_time
    print(clf.best_params_)


    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y, pred)*100,
                              "Time":     total_cross_val},
                              ignore_index=True)


# YOUR CODE HERE
# results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
# results_ord.index += 1
# results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

# print(results_ord)
# results_ord.to_csv("multtest.csv")
