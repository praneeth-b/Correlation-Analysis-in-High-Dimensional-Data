import numpy as np
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/praneeth/projects/sst/git/Correlation-Analysis-in-High-Dimensional-Data/multiple-datasets')
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis

n_sets = 4
signum = 4
tot_dims = 4
M = 200
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
                                                maxIters=99
                                                )

estimator.generate_structure()
estimator.run()

print("experiment complete")
