
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


train_data = "train.csv"
kaggle_test = "test.csv"
train_df = pd.read_csv(train_data)
kaggle_test_df = pd.read_csv(kaggle_test)

##### fills names with the title instead and categorises titles
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

##### Fill missing ages based on the title mean ages.
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

##### Apply the titles and missing ages.
pp_train_df = fill_missing_ages(train_df)
x = pp_train_df.drop(columns=["Survived", 'Ticket'])
y = pp_train_df['Survived']

x_train, x_val, y_train, y_val = train_test_split(x,y, random_state=909)

#####################
numeric_features = ['Age', 'Fare']
numeric_transformer = pipeline.Pipeline(steps=[
        ("imputer", impute.SimpleImputer(strategy="mean")),
        ('scaler',StandardScaler()),
])

categorical_features = ['Embarked', 'Pclass', 'Title', 'Sex']
categorical_transformer = pipeline.Pipeline(steps=[
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ])

preprocessor = compose.ColumnTransformer(transformers=[
    ("num_transform", numeric_transformer, numeric_features),
    ("cat_transform", categorical_transformer, categorical_features),
])


param_grid =[
    {
        "preprocessor__num_transform__imputer__strategy": ["mean", "median"],
        "nn_mlpc__solver": ['lbfgs', "sgd", "adam"],
        "svmSvc__kernel":["linear", "rbf"],
    }
    ]

mult_classifiers = {
        #"LM Linear Regression": LinearRegression(), # not useful for classification on titanic
        #"LM Logistic Regression": LogisticRegression(),
        #"LM Ridge": RidgeClassifier(),
        #"LM Lasso": Lasso(), Not useful for titanic dataset
        "nn_mlpc": MLPClassifier(random_state=909),
        #"SVM Linear": svm.SVC(kernel='linear'),
        "svmSvc": svm.SVC(kernel='rbf'),
        #"KNN": KNeighborsClassifier(),
        #"BM Guassian Naive Bayes": GaussianNB(),
        #"BM Multinominal Naive Bayes":MultinomialNB(),

}
pipedmodels = {model_name: pipeline.make_pipeline(preprocessor, model) for model_name, model in mult_classifiers.items()}

for pipe_name, pipe in pipedmodels.items():
    print(pipe_name)
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(x, y)
    print(search.best_params_)

# grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=10, verbose=1, n_jobs=-1)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_score_)
# print(grid_search.best_params_)
# pipeL.fit(x_train,y_train)
#
# score = pipeL.score(x_val,y_val)
# print(score)






















# AIMwpS
