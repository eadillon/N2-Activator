# N2-Activator

.__   __.  ___           ___       ______ .___________. __  ____    ____  ___   .___________.  ______   .______      
|  \ |  | |__ \         /   \     /      ||           ||  | \   \  /   / /   \  |           | /  __  \  |   _  \     
|   \|  |    ) |       /  ^  \   |  ,----'`---|  |----`|  |  \   \/   / /  ^  \ `---|  |----`|  |  |  | |  |_)  |    
|  . `  |   / /       /  /_\  \  |  |         |  |     |  |   \      / /  /_\  \    |  |     |  |  |  | |      /     
|  |\   |  / /_      /  _____  \ |  `----.    |  |     |  |    \    / /  _____  \   |  |     |  `--'  | |  |\  \----.
|__| \__| |____|    /__/     \__\ \______|    |__|     |__|     \__/ /__/     \__\  |__|      \______/  | _| `._____|
                                                                                                                     
This is a program that uses single crystal x-ray diffraction data of metal-dinitrogen complexes to make structural predictions about the activation of dinitrogen by compounds without known crystal structures.

Let me break down the pipeline of how this program works

The input for the program is a .cor file generated from Conquest, a software available from the Cambridge Crystallography Data Centre. The .cor file contains the atomic coordinates of the results of a search of their database. For our purposes, the .cor file will contain entries in which a metal-dinitrogen bond is present.

First, the .cor file is fed into the module "extraction.ipynb", which runs in a Jupyter notebook. Here, you can enter the .cor file.
The output will be three folders (training, test, and validation) into which the entries are sorted. In each of the entries, the dinitrogen fragment has been identified by the extraction program. This will be important later on. For your convenience, data produced by the extraction module are available folders are available in a .zip file.

Then, the program "run_it.ipynb" can be used to load these data into a neural network for training. To convert the entries into input vectors, the rewrap function is needed and will take two arguements, the data file and the desired length of input vector. For your convenience, the desired number of atoms can be entered above, and the corresponding input vector length will then be calculated. The input vectors will then contain four data points per atom: atomic number and 3D cartesian coordinates (x,y,z). The atomic number can be converted into more chemically meaningful descriptors (period, group, and electronegativity) using the AN_to_PC function.

Then, the network can be loaded from N2_neural_network. There are two options for the cost function, but for the best results, choose the MeanSquaredErrorCostWithConstraint.

Now the network can be trained using the SGD function in the Network class. There, you will need to enter the training data, number of epochs, batch size, learning rate, regularization parameter, learning rate decay, test data, respectfully, and indicated which network performance indicators you would like to use. More indicators lead to somewhat slower performance, but you'll find that the network trains fairly quickly regardless. Overall, training with 100 epochs should take no longer than a minute or two. The resutlls can be saved to a CSV file to produce a learning curve. The network can be saved by using the save function in the Network class.

To survey network heuristics, the grid search program is available. There, you enter the parameters in the params_grid. Once running, a text line should indicate approximately how long the grid search should take for the given parameters. A window will also pop up to track the progress in real time and keep track of the best accuracy and best parameters.

Once the network is trained, the position of dinitrogen from a given set of metal and ligand coordinated can be predicted in a single feedforward pass which should take no more than 100 ms. This is much faster than performing the analagous geometry optimization by DFT which can take anywhere from 30 minutes to multiple hours. To do so, the hallucinate_xyz function from visions2_xyz can be used. It takes the coordinates of interest as an input, with the metal where N2 is expectted to occur as the first line in the file. Then, additional required arguments are the saved network file, the vector length, and indicator if the AN_to_PC conversion was used for training the network. Then, the hallucinate program will produce an .xyz file with the predicted M-N2 complex.

For questions about using this program, please reach out to me via email at eadillon@caltech.edu
