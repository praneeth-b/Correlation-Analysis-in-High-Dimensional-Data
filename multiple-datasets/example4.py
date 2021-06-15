import numpy as np
from itertools import combinations
from MultisetDataGen import MultisetDataGen_CorrMeans
from E_val_E_vec_tests import Eval_Evec_test
from graph_visu import visualization
from metrics import Metrics
from CorrelationStructureGen import CorrelationStructureGen

# this is to simulate example-1 scene-1 of the matlab based repo!

n_sets = 5  # of data sets
signum = 4  # # of correlated + independent signals per set
tot_dims = 4  # # of sensors per set
M = 1000  # # of samples per set
num_iter = 1 * 1e1  # # of trials for each data point
SNR_vec = np.arange(-9, 16, 3)  # SNR vector ranging from -10 to 15dB
full_corr = 1  # # of signals correlated across all data sets
corr_across = [4, 3]  # across how many data sets should each additional signal be correlated?

RealComp = 'real'  # real/complex data  (only real is coded)

Distr = 'gaussian'  # gaussian or laplacian sources
sigmad = 2  # variance of correlated signals
sigmaf = 1  # variance of independent signals
mixing = 'orth'  # mixing matrix type ('orth'/'randn')
color = 'white'  # noise type ('white'/'colored')
MAcoeff = 1  # moving average coefficients for colored noise
ARcoeff = 1  # auto-regressive coefficients for colored noise
maxIters = 99  # maximum # of random draws allowed to find a positive definite covariance matrix

x_corrs = list(combinations(range(n_sets), 2))
if n_sets < 6:
    x_corrs = list(reversed(x_corrs))

subspace_dim = np.array([tot_dims] * n_sets)
tot_corr = np.append(np.tile(n_sets, [1, full_corr]), corr_across)
flag = 0
p_c = np.array([[0.7000, 0, 0.7000],
                [0.7000, 0.7000, 0],
                [0.7000, 0, 0],
                [0.7000, 0.7000, 0],
                [0.7000, 0, 0],
                [0.7000, 0.7000, 0],
                [0.7000, 0.7000, 0.7000],
                [0.7000, 0, 0.7000],
                [0.7000, 0.7000, 0],
                [0.7000, 0.7000, 0]])

p = np.append(p_c, np.zeros((len(x_corrs), signum - tot_corr.shape[0])), 1)
sigma_signals = sigmaf * np.ones(p.shape)
sigma_signals[np.where(p > 0)] = sigmad
n_combs = len(x_corrs)
corr_means = [.8, .7, .6]  # % mean of the correlation .7 .5 .6
# % coefficients of each signal for all
# % data sets
corr_std = [.1, .1, .1]
# if flag == 1:
#     corr_obj = CorrelationStructureGen(n_sets, tot_corr,
#                                        corr_means, corr_std, signum, sigmad, sigmaf, maxIters)
#     p, sigma_signals = corr_obj.generate(max_iters=1)
#
# # corr_obj = CorrelationStructureGen(n_sets, tot_corr,
# #                                    corr_means, corr_std, signum, sigmad, sigmaf, maxIters)
# # p2, sigma_signals2 = corr_obj.generate(max_iters=1)

sigmaN = .32
print("ready")

datagen = MultisetDataGen_CorrMeans(subspace_dim, signum, x_corrs, mixing, sigmad, sigmaf, sigmaN, color, n_sets, p,
                                    sigma_signals, M, MAcoeff, ARcoeff, Distr)
X, R, A, S = datagen.generate()
# print(X[0].shape)
corr_truth = np.zeros((n_combs, tot_dims))
idx_c = np.nonzero(p)
corr_truth[idx_c] = 1  # this is the ground truth correllation.
corr_truth = np.transpose(corr_truth)
# evaluate using Evec and Eval tests
Pfa_eval = 0.05
Pfa_evec = 0.05
B = 1000

corr_test = Eval_Evec_test(X, Pfa_eval, Pfa_evec, B)
corr_est, d_cap, u_struc = corr_test.find_structure()
# print (x_corrs)
perf = Metrics(corr_truth, corr_est)
precision, recall = perf.PrecisionRecall()

viz = visualization(corr_est, u_struc, x_corrs, signum, n_sets)
viz.visualize()
print("done")
