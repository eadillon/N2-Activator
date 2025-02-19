#import modules
import os
import shutil

#import specialized functions
from cheesewire import cheese_knife
from unCORker7_zaxis import unCORk
from bottle_service import pour
from drink import sip

## input .cor file from CCSD Conquest search with full path length

def wine_and_dine(file):
    #partition full .cor files into individual files for each structure
    cheese_knife(file)
    print("The cheese has been cut.")
    print("Now the wine can be unCORked.")
    
    # make new directory for .csv files
    path = os.path.dirname(os.path.abspath(file))
    folder_name = os.path.basename(file)[:-4]+"_csv"
    os.chdir(path)
    if os.path.exists(folder_name): #overwrite folder if it already exists
        shutil.rmtree(folder_name)
    os.mkdir(folder_name) #creates folder in path directory
    
    #convert raw .cor files into processed csv files
    os.chdir(os.path.basename(file)[:-4])
    wine_bottles = os.listdir(os.path.abspath(file)[:-4]) # open folder of .cor as wine_bottles
    for i in range(len(wine_bottles)):
        os.chdir(path+"/"+folder_name) # change directory to csv folder
        os.mkdir(wine_bottles[i][:-4]) # make entry folder
        os.chdir(path + "/" + os.path.basename(file)[:-4]) # change directory back to .cor folder
        # tag the percent completion and current entry
        percent = i / len(wine_bottles) * 100
        print("UnCORking " + wine_bottles[i] + f' - {percent:.1f}' +"% complete")
        unCORk(wine_bottles[i]) # unCORk into .cor folder
        file_name = wine_bottles[i][:-4]
        files = (file_name + "onlyN2.csv",file_name + "noN2.csv")
        dest = path + "/" + folder_name + "/" + wine_bottles[i][:-4]
        for f in range(len(files)):
            shutil.move(files[f],dest) # move files into folder in csv folder
        os.remove(wine_bottles[i])
    os.chdir(path)
    os.rmdir(os.path.basename(file)[:-4]) # remove temporary cheese_knife folder
    print("UnCORking 100% Complete")
    
    # randomly sort data into training, test, and validation sets
    # by default, 50 separate structures are sent into both test and validation
    pour(file)
    print("The wine has been poured into three folders.")
    
    # read .cor files as 1D input and output vectors
    training_data, test_data, validation_data = sip(file)
    print("Wining and dining has come to an end,")
    print("Thank you dearly you generous friend,")
    print("A debt so great I could never ammend,")
    print("A small token of thanks, these data I lend.")
    return training_data, test_data, validation_data