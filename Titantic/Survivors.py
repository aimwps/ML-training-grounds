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
kaggle_test = "test.csv"
train_df = pd.read_csv(train_data)
kaggle_test_df = pd.read_csv(kaggle_test)


### preprocessing

get_passanger_title  = lambda x: x.split(",")[1].split(".")[0].strip()
get_passanger_deck = lambda x: x[0] if isinstance(x, str) else "Z"
get_ticket_num = lambda x: "".join([i for i in x.split() if i.isnumeric()]).strip()

train_df['Ticket'] = train_df['Ticket'].map(get_ticket_num)
train_df['Title'] = train_df['Name'].map(get_passanger_title)
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
train_df['Title'] = train_df['Title'].map(title_dictionary)
age_by_title = train_df.groupby('Title').agg({"Age": np.mean}).reset_index()
print(age_by_title)
age = age_by_title.pivot(columns='Title', values='Age')
print(age.head(6))
