import numpy as np
import scipy.io
import sys

sys.path.insert(0, "/home/praneeth/projects/sst/git/Correlation-Analysis-in-High-Dimensional-Data/")
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis

# fetch audio data from file
mat = scipy.io.loadmat('dataset/audio_data.mat')
mat['Y_cell'][0][0].shape
d_cell = []

for i in range(10):
    cell = mat['Y_cell'][i][0]
    d_cell.append(cell)

n_sets = len(d_cell)
signum = d_cell[0].shape[0]
tot_dims = d_cell[0].shape[0]
M = d_cell[0].shape[1]
estimator = MultidimensionalCorrelationAnalysis(n_sets, signum,  M,

                                                simulation_data_type='real',  # synthetic / real
                                                Pfa_eval=0.05,
                                                Pfa_evec=0.05,
                                                threshold= 0.1, #1/np.sqrt(10),
                                                bootstrap_count=1000)
estimator.run(d_cell)

print("experiment complete")