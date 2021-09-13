import numpy as np
import sys
import os
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis



n_sets = 4  # number of datasets
signum = 5  # signals in each dataset
tot_dims = 5
M = 100  # data samples
estimator = MultidimensionalCorrelationAnalysis( n_sets, signum,
                                                 M,
                                                 tot_dims=signum,
                                                 num_iter=1,
                                                 #full_corr=1,
                                                 #corr_across=[3,2,],
                                                 #corr_means=[0.8],
                                                percentage_corr=True,
                                                corr_input=[100, 75],
                                                #corr_std=[0.1],
                                                #SNR_vec = [10]
                                                )

# synthetic data
estimator.generate_structure(disp_struc=True)
st, d_hat = estimator.run()

print("experiment complete")
