
import numpy as np
from correlation_analysis  import MultidimensionalCorrelationAnalysis

n_sets = 10
signum = 5
tot_dims = 10
M = 100
estimator = MultidimensionalCorrelationAnalysis(n_sets, signum, tot_dims, M,
                                                num_iter=1,
                                                full_corr=1,
                                                corr_across=[7, 6],
                                                corr_means=[0.8, 0.8, 0.8],
                                                corr_std=[0.01, 0.01, 0.01],
                                                RealComp='real',
                                                Distr='gaussian',
                                                sigmad=10,
                                                sigmaf=3,
                                                SNR_vec=[-2], #np.arange(-9, 16, 3),
                                                mixing='orth',
                                                color='white',
                                                MAcoeff=1,
                                                ARcoeff=1,
                                                maxIters=99
                                                )

estimator.generate_structure()
estimator.run()

print("experiment complete")