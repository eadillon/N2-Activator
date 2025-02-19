# the goal of this program is to convert the csv files from pour into 
# one-dimensional vectors
import os
import pandas as pd
import numpy as np

def vectorize_onlyN2(ID):
    df = pd.read_csv(ID+'onlyN2.csv',header=None) # input vector
    df_vectorized = np.zeros((6,1))
    for i in range(2):
        for j in range(3):
            df_vectorized[3*i+j] = df.iloc[i][j+1]
    return df_vectorized

def vectorize_noN2(ID):
    df = pd.read_csv(ID+'noN2.csv',header=None) # input vector
    df_vectorized = np.zeros(len(df)*4)
    for i in range(len(df)):
        for j in range(4):
            df_vectorized[4*i+j] = df.iloc[i][j]
    return df_vectorized

def sip(file):
    
    path = os.path.dirname(os.path.abspath(file))
    # move to training data folder
    os.chdir(path+"/"+os.path.basename(file)[:-4]+"_csv") # open training data folder
    training_list = os.listdir(path+"/"+os.path.basename(file)[:-4]+"_csv") # create list of folders contained inside
    training_data = [] # create list to store each vector set
    for i in range(len(training_list)): # for each, convert noN2 and onlyN2 into tuple with coor
        ID = training_list[i]
        os.chdir(path+"/"+os.path.basename(file)[:-4]+"_csv/"+ID)
        onlyN2 = vectorize_onlyN2(ID)
        noN2 = vectorize_noN2(ID)
        data_pt = (onlyN2,noN2)
        percent = i / len(training_list) * 100
        print("Sipping " + training_list[i] + f' - {percent:.1f}' +"% complete")
        training_data.append(data_pt)
    print("Sipping 100% complete. The training glass is empty.")
    
    # move to test data folder
    os.chdir(path+"/"+os.path.basename(file)[:-4]+"_testdata") # open training data folder
    test_list = os.listdir(path+"/"+os.path.basename(file)[:-4]+"_testdata") # create list of folders contained inside
    test_data = [] # create list to store each vector set
    for i in range(len(test_list)): # for each, convert noN2 and onlyN2 into tuple with coor
        ID = test_list[i]
        os.chdir(path+"/"+os.path.basename(file)[:-4]+"_testdata/"+ID)
        onlyN2 = vectorize_onlyN2(ID)
        noN2 = vectorize_noN2(ID)
        data_pt = (onlyN2,noN2)
        percent = i / len(test_list) * 100
        print("Sipping " + test_list[i] + f' - {percent:.1f}' +"% complete")
        test_data.append(data_pt)
    print("Sipping 100% complete. The test glass is empty.")

    # move to validation data folder
    os.chdir(path+"/"+os.path.basename(file)[:-4]+"_validationdata") # open training data folder
    validation_list = os.listdir(path+"/"+os.path.basename(file)[:-4]+"_validationdata") # create list of folders contained inside
    validation_data = [] # create list to store each vector set
    for i in range(len(validation_list)): # for each, convert noN2 and onlyN2 into tuple with coor
        ID = validation_list[i]
        os.chdir(path+"/"+os.path.basename(file)[:-4]+"_validationdata/"+ID)
        onlyN2 = vectorize_onlyN2(ID)
        noN2 = vectorize_noN2(ID)
        data_pt = (onlyN2,noN2)
        percent = i / len(validation_list) * 100
        print("Sipping " + validation_list[i] + f' - {percent:.1f}' +"% complete")
        validation_data.append(data_pt)
    print("Sipping 100% complete. The validation glass is empty.")
    print()
    print("Drinking complete. The whole bottle is empty!")
    return training_data, test_data, validation_data