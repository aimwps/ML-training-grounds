from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline

train_data = "train.csv"
test_data = "test.csv"
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)


def one_hot_encode_column(df, col_name):
    unique_splits = df[col_name].unique()
    print(unique_splits)
    df[[new_col for new_col in unique_splits]] = np.nan
    for new_col in unique_splits:
            df[new_col] = df[col_name].str.contains(new_col)
            df[new_col] = df[new_col].map({True: 1, False: 0})
    return df

def drop_nan_ehc_embarked(df):
    df = df[df['Embarked'].notna()]
    df = one_hot_encode_column(df, 'Embarked')
    df.rename(columns={'C':"Cherbourg", 'Q':"Queenstown", 'S':"Southampton"}, inplace=True)
    return df

def gender_to_int(df):
    df['Sex'] = df['Sex'].map({"male":0, "female":1})
    return df

def add_missing_fare(df):
    df.loc[(df.Fare == 0) & (df.Pclass == 1), "Fare"] = 84.19
    df.loc[(df.Fare == 0) & (df.Pclass == 2), "Fare"] = 20.66
    df.loc[(df.Fare == 0) & (df.Pclass == 3), "Fare"] = 13.68
    return df

def add_titles(df):
    df['Title'] = 'Unknown'
    for idx, row in df.iterrows():
    # bracket ="Unknown"
        if "Mr." in row['Name']:
            bracket = "Mr"
        elif "Miss." in row['Name']:
            bracket = "Miss"
        elif "Mrs." in row['Name']:
            bracket = "Mrs"
        elif "Master."in row['Name']:
            bracket = "Master"
        elif"Dr" in row['Name']:
            bracket = "Professional"
        elif"Rev" in row['Name']:
            bracket = "Professional"
        elif "Major"in row['Name']:
            bracket = "Professional"
        elif "Col" in row['Name']:
            bracket = "Professional"
        elif "Capt"in row['Name']:
            bracket = "Professional"
        elif "Countess" in row['Name']:
            bracket = "Professional"
        elif "Don" in row['Name']:
            bracket = "Professional"
        elif "Mme." in row['Name']:
            bracket= "Professional"
        elif "Mlle." in row['Name']:
            bracket = "Miss"
        elif "Jonkheer." in row["Name"]:
            bracket = "Master"
        elif "Ms." in row["Name"]:
            bracket = "Mr"
        elif "Sir." in row["Name"]:
            bracket = "Professional"
        elif "Lady." in row["Name"]:
            bracket = "Mrs"
        else:
            bracket ="Unknown"
        df.at[idx,'Title'] = bracket
    df = one_hot_encode_column(df, 'Title')
    return df

def add_missing_age(df):
    df.loc[(df.Age.isna()) & (df.Title == "Master"), "Age"] = 5
    df.loc[(df.Age.isna()) & (df.Title == "Miss"), "Age"] = 21
    df.loc[(df.Age.isna()) & (df.Title == "Mr"), "Age"] = 32
    df.loc[(df.Age.isna()) & (df.Title == "Mrs"), "Age"] = 35
    df.loc[(df.Age.isna()) & (df.Title == "Professional"), "Age"] = 4
    return df

def add_age_bracket_ehc(df):
    df['bracket'] = 'Unknown'
    for idx, row in df.iterrows():
        bracket ="Unknown"
        if row['Age'] <=1:
            bracket = "Infant"
        elif row['Age'] <5:
            bracket = "Toddler"
        elif row['Age'] <=12:
            bracket = "Child"
        elif row['Age'] <=19:
            bracket = "Teen"
        elif row['Age'] <=39:
            bracket = "Adult"
        elif row['Age'] <=59:
            bracket = "Middle"
        elif row['Age'] >=60:
            bracket = "Senior"
        else:
            bracket ="Unknown"
        df.at[idx,'bracket'] = bracket
    df = one_hot_encode_column(df, 'bracket')
    return df

def get_ml_columns(df):
    df = drop_nan_ehc_embarked(df)
    df = gender_to_int(df)
    df = add_missing_fare(df)
    df = add_titles(df)
    df = add_missing_age(df)
    df = add_age_bracket_ehc(df)
    df =  df.drop(['PassengerId','Embarked', 'Name','Cabin', 'Ticket', 'Title','Age', 'bracket'], axis=1)

    return df

test_data = get_ml_columns(test_df)
ml_data = get_ml_columns(train_df)
X = ml_data.drop('Survived', axis=1)
Y = ml_data['Survived']

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)


#### SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("\n")

### RFC MODEL
rfc = RandomForestClassifier(random_state=909).fit(x_train, y_train)
rfc_test_score = rfc.score(x_test,y_test)
print(f"Random Forest Classifier accuracy: {rfc_test_score}")
print("\n")


svc_settings = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
### Super Vector Machine Classifier
svc = svm.SVC(random_state=909).fit(x_train, y_train)
svc_test_score = svc.score(x_test, y_test)
print(f"SVC accuracy: {svc_test_score}")
print("\n")

### KNN
for i in range(3, 12,2):
    knn = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
    knn_test_score = knn.score(x_test, y_test)
    print(f"K nearest_neighbours neighbours= {i}/{11}: {knn_test_score}")
print("\n")

###Gaussian Model
gnb = GaussianNB().fit(x_train, y_train)
gnb_test_score = gnb.score(x_test, y_test)
print(f"Gaussian NB accuracy: {gnb_test_score}")
print("\n")


clf = make_pipeline(GaussianNB())
score = cross_val_score(clf, X, Y, cv=cv)
print(f"Cross Validated GNB {np.mean(score)}")
print("\n")


final_gnb = GaussianNB().fit(X, Y)
predictions = final_gnb.predict(test_data.notna())

test_df['Survived'] = predictions
submit = test_df[['PassengerId', 'Survived']]

print(submit.shape)
print(submit.head())
submit.to_csv("try_submit.csv", index=False)








### spacer
