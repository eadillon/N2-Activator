### takes in the coordinats from an .xyz file and returns the .xyz coordinates with the predicted N2 positions

#import libraries
import os
import numpy as np
import pandas as pd
#import special libraries/dictionaries
from sph2cart import sph2cart
from dabble4_xyz import dabble_xyz
from N2_neural_network import Network, load_network

def hallucinate_xyz(vector_file,network_file,length,pcoords=False): #input files with full path
    path = os.path.abspath(vector_file) #identify file path
    vector = dabble_xyz(vector_file,length=length,pcoords=pcoords) #format vector for feedforward
    net = load_network(network_file) # load network
    N2_out = net.feedforward(vector) # feedforward
    N2_out_flat = N2_out.flatten() #flatten to append values
    N2_atomlabels = np.insert(N2_out_flat, [0, int(len(N2_out_flat)*0.5)], 7) #insert atom coordinates into the correct positions
    N2 = N2_atomlabels.reshape(-1, 1) #reshape
    M = dabble_xyz(vector_file,length=length)
    MN2 = np.concatenate((N2, M), axis=0) # append N2 to coordinate file
    df = pd.DataFrame(MN2)
    reshaped_data = df.values.reshape(int(df.size/4), 4)
    reshaped_df = pd.DataFrame(reshaped_data, columns=['element', 'x', 'y', 'z'])
    reshaped_df['element'] = reshaped_df['element'].astype(int)
    os.chdir(os.path.dirname(vector_file))
    reshaped_df.to_csv(os.path.basename(vector_file)[:-4]+"_pred.xyz",index=False,header=False,sep=' ')