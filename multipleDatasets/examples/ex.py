import numpy as np
import sys
import os
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis
from multipleDatasets.visualization.graph_visu import visualization
import matplotlib.pyplot as plt



n_sets = 4  # number of datasets
signum = 5  # signals in each dataset
tot_dims = 5
M = 100  # data samples
estimator = MultidimensionalCorrelationAnalysis( n_sets, signum,
                                                 M,
                                                 tot_dims=signum,
                                                 num_iter=1,

                                                percentage_corr=True,
                                                corr_input=[100, 75],
                                                #corr_std=[0.1],
                                                SNR_vec = [10]
                                                )

# synthetic data
corr_truth = estimator.generate_structure(disp_struc=True)
corr_estimate, d_hat = estimator.run(disp_struc=False)
plt.ion()

viz = visualization(graph_matrix=corr_truth,  num_dataset=n_sets, label_edge=False)
viz.visualize("True Structure")
plt.ioff()
viz_op = visualization(graph_matrix=corr_estimate, num_dataset=n_sets, label_edge=False)
viz_op.visualize("Estimated_structure")

print("experiment complete")
