import numpy as np
import sys
import os

from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis

n_sets = 4  # number of datasets
signum = 5  # signals in each dataset
tot_dims = 8
M = 500  # data samples
estimator = MultidimensionalCorrelationAnalysis( n_sets, signum,
                                                 M,
                                                 tot_dims=signum,
                                                 num_iter=5,
                                                 full_corr=3,
                                                 corr_across=[2],
                                                 #corr_means=[0.8],
                                                percentage_corr=False,
                                                #corr_input=[100, 75],
                                                #corr_std=[0.1],
                                                SNR_vec = [-10, 2, 15]
                                                )
# synthetic data
estimator.generate_structure(disp_struc=False)
estimator.run()



print("experiment complete")
