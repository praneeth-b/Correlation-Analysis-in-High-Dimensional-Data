import numpy as np
import sys
import os

from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis

n_sets = 4   # number of datasets
signum = 4   # signals in each dataset
tot_dims = 4
M = 200      # data samples
estimator = MultidimensionalCorrelationAnalysis(n_sets, signum, tot_dims, M,
                                                num_iter=1,
                                                full_corr=1,
                                                corr_across=[2, 2],
                                                corr_means=[0.8, 0.8, 0.8],
                                                corr_std=[0.01, 0.01, 0.01],
                                                RealComp='real',
                                                Distr='gaussian',
                                                sigmad=10,
                                                sigmaf=3,
                                                SNR_vec=[-2],  # np.arange(-9, 16, 3),
                                                mixing='orth',
                                                color='white',
                                                MAcoeff=1,
                                                ARcoeff=1,
                                                maxIters=99,
                                                simulation_data_type='synthetic',  # synthetic / real
                                                Pfa_eval=0.05,
                                                Pfa_evec=0.05,
                                                bootstrap_count=1000
                                                )
# synthetic data
estimator.generate_structure()
estimator.run()

#to run with real data
# data = estimator.test_data_gen()
# estimator.run(data)

print("experiment complete")
