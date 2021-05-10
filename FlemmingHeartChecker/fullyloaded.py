import os.path
import pickle
import datetime as dt
import pandas as pd


#### Create a pipeline for testing different ML models

#### Check different uses of features

#### Check different  hyper parameters

#### Check if augumented data helps.



################################################################################
### Helper Functions

def no_overwrite_pickle(save_name, model):
    now_code = dt.datetime.now().strftime("%y%m%d%H%M")
    new_file_name = save_name + "_" + now_code + ".pickle"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = save_name + "_"  + now_code + f"_{str(duplicate_counter)}" + ".pickle"
        duplicate_counter += 1
    pickle.dump(model, open(new_file_name, 'wb'))

def no_overwrite_to_csv(save_name, dataframe):
    now_code = dt.datetime.now().strftime("%y%m%d%H%M")
    new_file_name = save_name + "_" + now_code + ".csv"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = save_name + "_"  + now_code + f"_{str(duplicate_counter)}" + ".csv"
        duplicate_counter += 1
    dataframe.to_csv(new_file_name)


df = pd.DataFrame({"A":[1,2,3], "B":[4,5,6]})

#test = TestClass("Mark", [1,2,3,4,5,6])

#no_overwrite_pickle("test", test)
no_overwrite_to_csv("dataframe", df)
