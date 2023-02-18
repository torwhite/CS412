# set up code for this experiment
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

#nfold

idx = np.random.permutation(num_examples).tolist() 
print(idx)