import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import tree


### Load the iris dataset, add target columns and clean column names
iris_data = datasets.load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df.columns =  iris_df.columns.str.replace(" (cm)", "", regex=False)
iris_df['target'] = iris_data.target
iris_df['class'] = iris_df['target'].map({0:"setosa" , 1:"versicolor" , 2:'virginica'})

iris_X = iris_df.drop(['target', "class"], axis=1)
iris_Y = iris_df['target']

# print(iris_Y.shape)
# print(iris_X.shape)

train_data, test_data, train_label, test_label = train_test_split(iris_X, iris_Y, test_size=0.2)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


class DecisionTree():
    def __init__(self, *args, **kwargs):
        self.clf = 
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None

    def load_split_data(self, data, labels):
        self.train_data, self.test_data, self.train_label, self.test_label = \
        train_test_split(iris_X, iris_Y, test_size=0.2)
        print(f"The training data(X) shape is : {self.train_data.shape}")
        print(f"The test data(X) shape is : {self.test_data.shape}")
        print(f"The training label(Y) shape is : {self.train_label.shape}")
        print(f"The test label(Y) shape is: {self.test_label.shape}")



test_tree = IrisDecisionTree()
test_tree.load_split_data(iris_X, iris_Y)
test_tree.fit(test_tree.train_data, test_tree.train_label)
tree.plot_tree(test_tree)
predict = test_tree.predict(test_tree.test_data)
print(predict)
