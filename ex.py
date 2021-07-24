import numpy as np
import sys
import os

from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis

n_sets = 4  # number of datasets
signum = 4  # signals in each dataset
tot_dims = 4
M = 1000  # data samples
estimator = MultidimensionalCorrelationAnalysis( n_sets, signum,
                                                 M,
                                                 tot_dims=signum,
                                                 num_iter=10,
                                                 #full_corr=1,
                                                 #corr_across=[2, 1],
                                                # # corr_means=[0.8, 0.8, 0.8],
                                                percentage_corr=True,
                                                corr_input=[100, 75, 50],
                                                # corr_std=[0.01, 0.01, 0.01],

                                                )
# synthetic data
estimator.generate_structure(disp_struc=True)
estimator.run()



print("experiment complete")
