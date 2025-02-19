#this version of the unCORker utilizes a normalized coodinate system

#import libraries
import io
import os
import pandas as pd
import numpy as np

#import special libraries
from cartestian_spherical_conversion import cart2sph
from N2search import findN2
from metal_center_search import findTM
from hydride_search import findhydride
from element_to_number import elem2num
from sph2cart import sph2cart
from zaxis import rotate_to_align_z,midpoint

def unCORk(file): # file with absolute path included
    
    ##create temporary file which gets deleted
    #define path for new file to be created in
    path = os.path.dirname(os.path.abspath(file))
    os.chdir(path)
    #create new file
    with io.open('temp.txt','w', encoding = 'windows-1252') as fp:
        pass
    #delete headers
    with io.open(file, "r", encoding = 'windows-1252') as input:
        with io.open(path+"/temp.txt", "w",encoding = 'windows-1252') as output:
            for line in input:
                if "**FRAG**" not in line.strip("\n"):
                    output.write(line)
    os.replace('temp.txt', file)
    input.close() #close .cor file
    fp.close() #close temporary file
    #read .cor files as csv
    table = pd.read_csv(file, sep='\s+', header=None)
    #delete index column
    table2 = table.drop(columns=[4])
    # name columns for indexing
    table2.columns = ['element','x','y','z']
    #delete rows containing disordered elements
    table2 = table2[table2["element"].str.contains('?', regex=False) == False]
    table_hold = table2.reset_index()
    table3 = table_hold.drop(columns="index")
    # rename rows ending in strings
    for j in range(len(table3)):
        c = table3.iloc[j,0] 
        if isinstance(c[-1], str):
            table3.iloc[j,0] = c[:-1]
    # delete numbers associated with elements
    for j in range(len(table3)):
        table3.iloc[j,0] = ''.join([i for i in table3.iloc[j,0] if not i.isdigit()])
    #quantify elements
    for j in range(len(table3)):
        table3.iloc[j,0] = elem2num(table3.iloc[j,0])
    # N2 search
    N2_ID = findN2(table3)
    # metal center search
    metal_ID = findTM(table3,N2_ID)
    #hydride search
    if any(table3.iloc[:,0] == 1):
        hydride_ID = findhydride(table3,metal_ID)
    else:
        hydride_ID = pd.DataFrame()
    #store hydride, N2, and metal data
    if hydride_ID.size >=1:
        table_H = table3.iloc[hydride_ID.index]
    else:
        table_H = pd.DataFrame()
    table_M = table3.iloc[metal_ID]
    #delete M
    table3 = table3.drop(metal_ID)
    #delete hydrogen atoms
    table3 = table3.drop(table3[table3["element"]==1].index)
    #add hydrides 
    if table_H.size>0:
        table3 = pd.concat([table_H,table3])
    #add M
    table3 = pd.concat([table_M,table3])
    #define metal center as origin
    origin_x = table3.iloc[0,1]
    origin_y = table3.iloc[0,2]
    origin_z = table3.iloc[0,3]
    for i in range(len(table3)):
        table3.iloc[i,1] = table3.iloc[i,1]-origin_x # shift x
        table3.iloc[i,2] = table3.iloc[i,2]-origin_y # shift y
        table3.iloc[i,3] = table3.iloc[i,3]-origin_z # shift z
    # convert cartesian to spherical coordinates
    table3[['r', 'phi', 'theta']] = table3.apply(lambda row: pd.Series(cart2sph(row['x'], row['y'], row['z'])), axis=1)
    table3 = table3.drop(['x','y','z'],axis=1)
    # sort by radius
    table4 = table3.sort_values(by='r')
    # convert to spherical back to cartesian coordinates
    table4[['x', 'y', 'z']] = table4.apply(lambda row: pd.Series(sph2cart(row['r'], row['phi'], row['theta'])), axis=1)
    table4 = table4.drop(['r','phi','theta'],axis=1)
    # rotate coordinates to align with the z-axis with the N2 midpoint
    mp = midpoint(table4[table4.index.isin(N2_ID)])
    table5 = rotate_to_align_z(table4,mp)
    # generate test N2
    table_onlyN2 = table5[table5.index.isin(N2_ID)]
    test_onlyN2_file = table_onlyN2.to_csv(file[len(file)-10:-4]+"onlyN2.csv", header=False,index=False)
    # generate test complex
    table_noN2 = table5[~table5.index.isin(N2_ID)]
    test_noN2_file = table_noN2.to_csv(file[len(file)-10:-4]+"noN2.csv", header=False,index=False)
    return test_noN2_file, test_onlyN2_file