"""The goal of this function is to take a .cor file from
a CCSD search and parse it into individual .cor files
"""
import os
import shutil

def cheese_knife(file):
    
    #create folder for .cor files
    directory = file[:-4] #name folder after input file name
    parent_dir = os.path.abspath(file) #identify file path
    l = len(file)
    parent_dir = parent_dir[:-l-1]
    path = os.path.join(parent_dir,directory)
    if os.path.exists(path): #overwrite folder if it already exists
        shutil.rmtree(path)
    os.mkdir(path)

    #identify lines containing **FRAG** and read names
    with open(file,"r") as input:
        os.chdir(path)
        for line in input:
            if "**FRAG**" in line.strip("\n"):
                str = line.strip("\n")
                filename = str[0:6] + ".cor"
                if os.path.exists(filename) == True:
                    filename = filename[0:6] + "_dup.cor"
                else:
                    newfile = open(filename,"x")
                    newfile.write(line)
            elif "**FRAG**" not in line.strip("\n"):
                    newfile.write(line)
    return directory