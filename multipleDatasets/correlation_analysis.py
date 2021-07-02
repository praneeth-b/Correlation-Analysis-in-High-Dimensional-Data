import numpy as np
from itertools import combinations
from multipleDatasets.simulateData.MultisetDataGen import MultisetDataGen_CorrMeans
from multipleDatasets.algo.E_val_E_vec_tests import Eval_Evec_test
from multipleDatasets.visualization.graph_visu import visualization
from multipleDatasets.simulateData.CorrelationStructureGen import CorrelationStructureGen
from multipleDatasets.metrics.metrics import Metrics
import matplotlib.pyplot as plt


class MultidimensionalCorrelationAnalysis:
    """
    Wrapper class to run multidimentional correlation analysis.
    """

    def __init__(self, n_sets, signum, tot_dims, M, **kwargs):
        """

        Args:
            n_sets (int):  number of datasets
            signum (int): number of signals in each dataset
            tot_dims (int): signal dimensions
            M (int): number of data samples
            **kwargs ():
        """
        self.M = M
        self.tot_dims = tot_dims
        self.signum = signum
        self.n_sets = n_sets
        self.param = kwargs

        self.x_corrs = list(combinations(range(n_sets), 2))
        if n_sets < 6:
            self.x_corrs = list(reversed(self.x_corrs))
        self.n_combs = len(self.x_corrs)
        self.subspace_dim = np.array([tot_dims] * n_sets)

    def generate_structure(self):
        """
        Generate the correlation structure for synthetic data given the full correlation and corr_across parameters
        Returns:

        """
        try:
            if any(y < 2 for y in self.param['corr_across']):
                raise ValueError("Minimum value of corr_across = 2, i.e. atleast 1 pair of datasets ")

        except ValueError as ve:
            print(ve)

        tot_corr = np.append(np.tile(self.n_sets, [1, self.param['full_corr']]), self.param['corr_across'])

        corr_obj = CorrelationStructureGen(self.n_sets, tot_corr,
                                           self.param['corr_means'], self.param['corr_std'], self.signum,
                                           self.param['sigmad'], self.param['sigmaf'], self.param['maxIters'])

        attempts = 4
        ans = "n"
        u_struc = 0
        while ans != "y":  # or attempts > 4:
            self.p, self.sigma_signals, self.R = corr_obj.generate()

            corr_truth = np.zeros((self.n_combs, self.tot_dims))
            idx_c = np.nonzero(self.p)
            corr_truth[idx_c] = 1  # this is the ground truth correllation.
            self.corr_truth = np.transpose(corr_truth)
            # visualize input correllation structure
            viz = visualization(np.transpose(self.p), u_struc, self.x_corrs, self.signum, self.n_sets)
            viz.visualize("Generated corr structure")
            ans = input("Continue with generated correlation structure?: y/n")

    def run(self):
        prec_vec = []
        rec_vec = []
        for snr in self.param['SNR_vec']:
            sigmaN = self.param['sigmad'] / (10 ** (0.1 * snr))
            print("SNR val = ", snr, " and sigmaN=", sigmaN)
            precision = 0
            recall = 0
            for i in range(self.param['num_iter']):
                print("iteration = ", i)

                datagen = MultisetDataGen_CorrMeans(self.subspace_dim, self.signum, self.x_corrs, self.param['mixing'],
                                                    self.param['sigmad'], self.param['sigmaf'], sigmaN,
                                                    self.param['color'],
                                                    self.n_sets, self.p,
                                                    self.sigma_signals, self.M, self.param['MAcoeff'],
                                                    self.param['ARcoeff'], self.param['Distr'], self.R)
                X, R, A, S = datagen.generate()

                # evaluate using Evec and Eval tests
                Pfa_eval = 0.05
                Pfa_evec = 0.05
                B = 1000

                corr_test = Eval_Evec_test(X, Pfa_eval, Pfa_evec, B)
                corr_est, d_cap, u_struc = corr_test.find_structure()
                perf = Metrics(self.corr_truth, corr_est)
                pr, re = perf.PrecisionRecall()
                precision += pr
                recall += re

            prec_vec.append(precision / self.param['num_iter'])
            rec_vec.append(recall / self.param['num_iter'])

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Precion recall plots for various SNR')
        ax1.plot(self.param['SNR_vec'], prec_vec, 'o-')
        ax1.set_ylabel('Precision')
        ax1.set_xlabel('SNR')

        ax2.plot(self.param['SNR_vec'], rec_vec, 'o-')
        ax2.set_ylabel('Recall')
        ax2.set_xlabel('SNR')
        plt.show()
        viz = visualization(self.corr_truth, u_struc, self.x_corrs, self.signum, self.n_sets)
        viz.visualize("True Structure")
        plt.ioff()
        viz_op = visualization(corr_est, u_struc, self.x_corrs, self.signum, self.n_sets)
        viz_op.visualize("Estimated_structure")

        print("done")
