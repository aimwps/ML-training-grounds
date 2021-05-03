import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder




########################################################################################################################################
#### Iris Data Set create data frame
iris_data = datasets.load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df.columns =  iris_df.columns.str.replace(" (cm)", "", regex=False)
iris_df['target'] = iris_data.target
iris_df['class'] = iris_df['target'].map({0:"setosa" , 1:"versicolor" , 2:'virginica'})
iris_X = iris_df.drop(['target', "class"], axis=1)
iris_Y = iris_df['target']
itrain_data, itest_data, itrain_label, itest_label = train_test_split(iris_X, iris_Y, test_size=0.2)

iris_clf = tree.DecisionTreeClassifier()
iris_clf.fit(itrain_data, itrain_label)
itext_tree = tree.export_text(iris_clf)
predict = iris_clf.predict(itest_data)
print(predict)

########################################################################################################################################
### Zoo Data create the dataframe
zoo_df = pd.read_csv("zoo.data")
zoo_df.columns = ["animal name"," hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone",
                    "breathes","venomous","fins","legs","tail","domestic","catsize","type"]
zoo_df.columns = zoo_df.columns.str.strip()
zoo_X = zoo_df.drop(['predator', 'animal name'], axis=1)
zoo_Y = zoo_df['predator']

### Create the model
ztrain_data, ztest_data, ztrain_label, ztest_label = train_test_split(zoo_X, zoo_Y, test_size=0.2)

def zoo_random_forest(max_depth_iterator):
    for i in range(1,max_depth_iterator):
        print(f"Max Depth = {i}")
        zoo_rfc = RandomForestClassifier(max_depth=i)
        zoo_rfc.fit(ztrain_data, ztrain_label)
        zoo_predict = zoo_rfc.predict(ztest_data)
        zoo_score = zoo_rfc.score(ztest_data, ztest_label)
        print(zoo_score)



# zoo_random_forest(1000)


########################################################################################################################################
### Car Data Set create the dataframe
car_df = pd.read_csv("car.data")
car_df.columns = ["buying","maint", "doors", "persons", "lug_boot", "safety", "class"]
car_X = car_df.drop('class', axis=1)
car_Y = car_df['class']
ctrain_data, ctest_data, ctrain_label, ctest_label = train_test_split(car_X, car_Y, test_size=0.2, random_state=0)

def car_random_forest(max_depth_iterator, min_split_iterator):
    best_score = (0,0,0)
    for i in range(1,max_depth_iterator+1):
        for ii in range(2,min_split_iterator+2):
            print(f"Max Depth = {i}, Min Split = {ii}")
            ehc = OneHotEncoder()
            ehc.fit(ctrain_data)
            ehc_ctrain_data = ehc.transform(ctrain_data).toarray()
            rfc = RandomForestClassifier(max_depth=i, min_samples_split=ii, random_state=0)
            rfc.fit(ehc_ctrain_data, ctrain_label)
            predict = rfc.predict(ehc.fit_transform(ctest_data).toarray())
            score = rfc.score(ehc.fit_transform(ctest_data).toarray(), ctest_label)
            print(f"The score is {score}")
            if score > best_score[0]:
                best_score = (score, i, ii)
                print("New best score!")
    print(f"The best score was {best_score[0]} from a max_depth of {best_score[1]} and a min split of {best_score[2]}")


car_random_forest(50, 5)

print(car_df.sample(10))
