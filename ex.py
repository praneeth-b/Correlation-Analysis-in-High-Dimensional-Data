import numpy as np
import sys
import os
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis


def bool_to_int(x):
    y=0
    for i,j in enumerate(x):
        y += j<<i
    return y


n_sets = 5  # number of datasets
signum = 4  # signals in each dataset
tot_dims = 3
M = 500  # data samples
estimator = MultidimensionalCorrelationAnalysis( n_sets, signum,
                                                 M,
                                                 tot_dims=signum,
                                                 num_iter=1,
                                                 full_corr=1,
                                                 corr_across=[3,2,],
                                                 #corr_means=[0.8],
                                                percentage_corr=False,
                                                #corr_input=[100, 75],
                                                #corr_std=[0.1],
                                                SNR_vec = [10]
                                                )

# synthetic data
estimator.generate_structure(disp_struc=False)
st, d_hat = estimator.run()
# arr = st[1,:]
# n = bool_to_int(arr.astype(np.int32))
# print(n)


print("experiment complete")
