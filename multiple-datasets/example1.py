import numpy as np
from itertools import combinations
from MultisetDataGen import MultisetDataGen_CorrMeans
from E_val_E_vec_tests import Eval_Evec_test
from graph_visu import visualization

# this is to simulate example-1 scene-1 of the matlab based repo!

n_sets = 4  # of data sets
signum = 7  # # of correlated + independent signals per set
tot_dims = 7  # # of sensors per set
M = 350  # # of samples per set
num_iter = 1 * 1e1  # # of trials for each data point
SNR_vec = np.arange(-10, 15, 3)  # SNR vector ranging from -10 to 15dB
full_corr = 3  # # of signals correlated across all data sets
corr_across = []  # across how many data sets should each additional signal be correlated?

RealComp = 'real'  # real/complex data  (only real is coded)

Distr = 'gaussian'  # gaussian or laplacian sources
sigmad = 1  # variance of correlated signals
sigmaf = 1  # variance of independent signals
mixing = 'orth'  # mixing matrix type ('orth'/'randn')
color = 'white'  # noise type ('white'/'colored')
MAcoeff = [1]  # moving average coefficients for colored noise
ARcoeff = [1]  # auto-regressive coefficients for colored noise
maxIters = 99  # maximum # of random draws allowed to find a positive definite covariance matrix


x_corrs = list(combinations(range(n_sets), 2))
subspace_dim = np.array([tot_dims] * n_sets)
tot_corr = np.append(np.tile(n_sets, [1, full_corr]), corr_across)

p_c = np.array([[0.9186, 0.9104, 0.6234],
                [0.6442, 0.8203, 0.7156],
                [0.8199, 0.7156, 0.5701],
                [0.6939, 0.7475, 0.7206],
                [0.7890, 0.6793, 0.8181],
                [0.6328, 0.6209, 0.8490]])

p = np.append(p_c, np.zeros((len(x_corrs), signum - tot_corr.shape[0])), 1)
sigma_signals = sigmaf * np.ones(p.shape)
sigma_signals[np.where(p > 0)] = sigmad
n_combs = len(x_corrs)

## add a loop here for iterating over snr vec

sigmaN = 0.0316
print("ready")
## add a loop to iterate over num_iter

datagen = MultisetDataGen_CorrMeans(subspace_dim, signum, x_corrs, mixing, sigmad, sigmaf, sigmaN, color, n_sets, p,
                                    sigma_signals, M, MAcoeff, ARcoeff, Distr)
X, R, A, S = datagen.generate()
print(X[0].shape)
corr_truth = np.zeros((n_combs, tot_dims))
idx_c = np.nonzero(p)
corr_truth[idx_c] = 1  # this is the ground truth correllation.

# evaluate using Evec and Eval tests
Pfa_eval = 0.05
Pfa_evec = 0.05
B = 1000

corr_est, d_cap = Eval_Evec_test(X, Pfa_eval, Pfa_evec, B).find_structure()
print (x_corrs)

viz = visualization(corr_est, x_corrs, signum, n_sets)
viz.visualize()
print("done")
