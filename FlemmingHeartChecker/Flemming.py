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
import pickle
import random


adf = pd.read_csv('augumented_dataset.csv')
df = pd.read_csv('heart.csv')

num_vars = ['age','trestbps','chol','thalach','oldpeak']
cat_vars = ['cp','slope','thal','sex','exang','ca','fbs','restecg']

cat_mult_pipeline = pipeline.Pipeline(steps=[
    ('One Hot', OneHotEncoder(handle_unknown="ignore")),
    ])

## For Numerical multipicative models
num_mult_pipeline = pipeline.Pipeline(steps=[
    ("Quantile transformer", StandardScaler()),
    ])

## For bringing nultiplicative numerical / Categorical together
mult_prepro = compose.ColumnTransformer(transformers=[
    ('multnum', num_mult_pipeline, num_vars),
    ('multcat', cat_mult_pipeline, cat_vars),
    ], remainder='passthrough')



OESTIMATOR = pipeline.make_pipeline(mult_prepro, KNeighborsClassifier(8))
X = df.drop(['target'], axis=1)
Y = df['target']
print("ORIGINAL DATASET")
#OESTIMATOR.fit(X,Y)
o_results = cross_val_score(OESTIMATOR,
                                 X, Y,
                                 cv=10,
                                 scoring="accuracy",
                                 n_jobs=-1)

print(f"The averaimwpsage original cross validated score is {np.mean(o_results)}")
# fn= "osaved-model.pickle"
# pickle.dump(OESTIMATOR, open(fn, 'wb'))


AESTIMATOR = pipeline.make_pipeline(mult_prepro, KNeighborsClassifier(8))
AX = adf.drop('target', axis=1)
AY = adf['target']

#AESTIMATOR.fit(AX,AY)

a_results = cross_val_score(AESTIMATOR,
                                 AX, AY,
                                 cv=10,
                                 scoring="accuracy",
                                 n_jobs=-1)


print(f"The average Augmented cross validated score is {np.mean(a_results)}")
# filename = "asaved-model.pickle"
# pickle.dump(AESTIMATOR, open(filename, 'wb'))



def make_noisey(data_point):
    noise_percent = 1 + random.randint(-5,5)/100
    return int(data_point * noise_percent)


def add_noise(samples):
    #noisy_samples = pd.DataFrame()
    #percent_noise = random.randint(1,10)
    #print(percent_noise)
    noisable_columns = ["age", "trestbps", "chol", "thalach"]
    for column in noisable_columns:
        #print(column)
        #print(samples[column].values)
        noisy_column = []
        print(f">>Applying noise to {column}")
        for item in samples[column].values:
            noisy_column.append(make_noisey(item))
        samples[column] = noisy_column

    #print(samples)
    return samples

def generate_data(dataset):
    generation_data = dataset.sample(int(len(dataset) / 2))
    noisy_data = add_noise(generation_data)
    dataset = dataset.append(noisy_data)
    return dataset

# new_ds = generate_data(df)
# new_ds.to_csv("augumented_dataset.csv")
