import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from MultisetDataGen import MultisetDataGen_CorrMeans
from E_val_E_vec_tests import Eval_Evec_test
from graph_visu import visualization
from CorrelationStructureGen import CorrelationStructureGen
from metrics import Metrics

n_sets = 4
signum = 5
tot_dims = 5
M = 500
num_iter = 1
full_corr = 1
corr_across = [4,3,2]

corr_means = [0.8, 0.8, 0.8, 0.8]
corr_std = [0.1, 0.1, 0.1, 0.1]
# % of each signal for all data sets
RealComp = 'real'
Distr = 'gaussian'
sigmad = 10
sigmaf = 3
#sigmaN = 10
SNR_vec = [10] #np.arange(-9, 16, 3)  # % SNR vector ranging from -10 to 15dB
mixing = 'orth'
color = 'white'
MAcoeff = 1
ARcoeff = 1
maxIters = 99

x_corrs = list(combinations(range(n_sets), 2))
if n_sets < 6:
    x_corrs = list(reversed(x_corrs))
n_combs = len(x_corrs)
subspace_dim = np.array([tot_dims] * n_sets)
tot_corr = np.append(np.tile(n_sets, [1, full_corr]), corr_across)

corr_obj = CorrelationStructureGen(n_sets, tot_corr,
                                   corr_means, corr_std, signum, sigmad, sigmaf, maxIters)
attempts = 4
ans = "n"
u_struc=0
while ans != "y" : #or attempts > 4:
    p, sigma_signals, R = corr_obj.generate()

    corr_truth = np.zeros((n_combs, tot_dims))
    idx_c = np.nonzero(p)
    corr_truth[idx_c] = 1  # this is the ground truth correllation.
    corr_truth = np.transpose(corr_truth)
    # visualize input correllation structure
    viz = visualization(corr_truth, u_struc, x_corrs, signum, n_sets)
    viz.visualize()
    ans = input("Continue with generated correlation structure?: y/n" )



prec_vec = []
rec_vec = []
for snr in SNR_vec:
    sigmaN = sigmad/(10 ** (0.1*snr))
    print("SNR val = ", snr, " and sigmaN=", sigmaN)
    precision = 0
    recall = 0
    for i in range(num_iter):
        print("iteration = ", i)

        datagen = MultisetDataGen_CorrMeans(subspace_dim, signum, x_corrs, mixing, sigmad, sigmaf, sigmaN, color,
                                            n_sets, p,
                                            sigma_signals, M, MAcoeff, ARcoeff, Distr, R)
        X, R, A, S = datagen.generate()

        # evaluate using Evec and Eval tests
        Pfa_eval = 0.05
        Pfa_evec = 0.05
        B = 1000

        corr_test = Eval_Evec_test(X, Pfa_eval, Pfa_evec, B)
        corr_est, d_cap, u_struc = corr_test.find_structure()
        perf = Metrics(corr_truth, corr_est)
        pr, re = perf.PrecisionRecall()
        precision += pr
        recall += re

    prec_vec.append(precision/num_iter)
    rec_vec.append(recall/num_iter)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Precion recall plots for various SNR')
ax1.plot(SNR_vec, prec_vec, 'o-')
ax1.set_ylabel('Precision')
ax1.set_xlabel('SNR')

ax2.plot(SNR_vec, rec_vec, 'o-')
ax2.set_ylabel('Recall')
ax2.set_xlabel('SNR')
plt.show()

plt.ioff()
viz_op = visualization(corr_est, u_struc, x_corrs, signum, n_sets)
viz_op.visualize()

print("done")