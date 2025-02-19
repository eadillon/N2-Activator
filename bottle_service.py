import os, random, shutil

def pour(file):
    path = os.path.dirname(os.path.abspath(file))
    os.chdir(path)

    # make test data folder
    test_folder = os.path.basename(file)[:-4]+"_testdata"
    if os.path.exists(test_folder): #overwrite folder if it already exists
        shutil.rmtree(test_folder)
    os.mkdir(test_folder)
    # make validation data folder
    validation_folder = os.path.basename(file)[:-4]+"_validationdata"
    if os.path.exists(validation_folder): #overwrite folder if it already exists
        shutil.rmtree(validation_folder)
    os.mkdir(validation_folder)

    #identify source folder
    csv_folder = os.path.abspath(file)[:-4]+"_csv" # source folder

    for i in range(50): # move 50 items to training data folder
        random_file=random.choice(os.listdir(csv_folder))
        source_file="%s/%s"%(csv_folder,random_file)
        shutil.move(source_file,test_folder)
    for i in range(50): # move 50 items to validation data folder
        random_file=random.choice(os.listdir(csv_folder))
        source_file="%s/%s"%(csv_folder,random_file)
        shutil.move(source_file,validation_folder)
    return