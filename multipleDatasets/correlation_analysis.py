import numpy as np
from itertools import combinations
from multipleDatasets.simulateData.MultisetDataGen import MultisetDataGen_CorrMeans
from multipleDatasets.algo.E_val_E_vec_tests import Eval_Evec_test
from multipleDatasets.visualization.graph_visu import visualization
from multipleDatasets.simulateData.CorrelationStructureGen import CorrelationStructureGen
from multipleDatasets.metrics.metrics import Metrics
import matplotlib.pyplot as plt
from multipleDatasets.utils.helper import get_default_params

default_params = {
    'n': 4,
    'signum': 4,
    'M': 300,
    'num_iter': 1,
    'percentage_corr': True,
    'RealComp': 'real',
    'Distr': 'gaussian',
    'sigmad': 10,
    'sigmaf': 3,
    'SNR_vec': [15],
    'mixing': 'orth',
    'color': 'white',
    'MAcoeff': 1,
    'ARcoeff': 1,
    'maxIters': 99,
    'simulation_data_type': 'synthetic',  # synthetic / real
    'Pfa_eval': 0.05,
    'Pfa_evec': 0.05,
    'bootstrap_count': 1000,
    'threshold': 0
}


class MultidimensionalCorrelationAnalysis:
    """
    Wrapper class to run multidimentional correlation analysis.
    """

    def __init__(self, n_sets=default_params['n'], signum=default_params['signum'], M=default_params['M'], **kwargs):
        """

        Args:
            n_sets (int):  number of datasets
            signum (int): number of signals in each dataset
            tot_dims (int): signal dimensions
            M (int): number of data samples
            **kwargs (): Other key word arguements to be passed to further objects.
        """
        self.M = M
        self.signum = signum
        self.n_sets = n_sets
        self.param = kwargs
        if 'tot_dims' not in self.param:
            self.param['tot_dims'] = self.signum
        self.tot_dims = self.param['tot_dims']

        self.x_corrs = list(combinations(range(n_sets), 2))
        if n_sets < 6:
            self.x_corrs = list(reversed(self.x_corrs))
        self.n_combs = len(self.x_corrs)
        self.subspace_dim = np.array([self.param['tot_dims']] * n_sets)
        self.synthetic_structure = False
        try:
            self.simulation_data_type = self.param['simulation_data_type']
        except KeyError:
            self.simulation_data_type = 'synthetic'

        self.param = get_default_params(self.param, default_params)


    def calc_input_corr_vals_from_percent(self, corr_list):
        # check array len < signum
        full_corr = 0
        corr_accross = []
        corr_list.sort(reverse=True)
        for ele in corr_list:
            if ele == 100:
                full_corr += 1

            else:
                ca = int((ele / 100) * self.n_sets)
                corr_accross.append(ca)
        return full_corr, corr_accross

    def generate_structure(self, disp_struc=False):
        """

        """

        if 'percentage_corr' not in self.param:
            raise Exception("percentage_corr must be set to True or False")
        if self.param['percentage_corr']:
            assert 'corr_input' in self.param, "correlations between signals must be input as a list of percentages"
            self.param['full_corr'], self.param['corr_across'] = self.calc_input_corr_vals_from_percent(
                self.param['corr_input'])

            if 'corr_means' not in self.param and 'corr_std' not in self.param:

                self.param['corr_means'] = [0.8] * len(self.param['corr_input'])
                self.param['corr_std'] = [0.1] * len(self.param['corr_input'])


        else:
            assert 'full_corr' and 'corr_across' in self.param , " correlation structure of synthetic data must be provided"
            if 'corr_means' not in self.param:
                self.param['corr_means'] = [0.8]*(self.param['full_corr'] + len(self.param['corr_across']))
            if 'corr_std' not in self.param:
                self.param['corr_std'] = [0.1]*(self.param['full_corr'] + len(self.param['corr_across']))

        try:
            if any(y < 2 for y in self.param['corr_across']):
                raise ValueError("Minimum value of corr_across = 2, i.e. atleast 1 pair of datasets ")

        except ValueError as ve:
            print(ve)

        tot_corr = np.append(np.tile(self.n_sets, [1, self.param['full_corr']]), self.param['corr_across'])


        corr_obj = CorrelationStructureGen(self.n_sets, tot_corr,
                                           self.param['corr_means'], self.param['corr_std'], self.signum,
                                           self.param['sigmad'], self.param['sigmaf'], self.param['maxIters'])

        attempts = 0
        ans = "n"
        u_struc = 0

        while ans != "y" or attempts > 4:
            self.p, self.sigma_signals, self.R = corr_obj.generate()

            corr_truth = np.zeros((self.n_combs, self.tot_dims))
            idx_c = np.nonzero(self.p)
            corr_truth[idx_c] = 1  # this is the ground truth correllation.
            self.corr_truth = np.transpose(corr_truth)
            # visualize input correllation structure
            if disp_struc:
                viz = visualization(np.transpose(self.p), u_struc, self.x_corrs, self.signum, self.n_sets)
                viz.visualize("Generated corr structure")
                ans = input("Continue with generated correlation structure?: y/n")

            else:
                break
            attempts += 1
        self.synthetic_structure = True

    def test_data_gen(self, ):
        sigmaN = 1
        datagen = MultisetDataGen_CorrMeans(self.subspace_dim, self.signum, self.x_corrs, self.param['mixing'],
                                            self.param['sigmad'], self.param['sigmaf'], sigmaN,
                                            self.param['color'],
                                            self.n_sets, self.p,
                                            self.sigma_signals, self.M, self.param['MAcoeff'],
                                            self.param['ARcoeff'], self.param['Distr'], self.R)
        X, R, A, S = datagen.generate()
        return X

    def run_syntheticData(self):
        """
        Simulates data for the given correlation sturcture and runs the eval and evec tests on the data to estimate
        the correlation structure Returns:

        """
        assert self.synthetic_structure, " Call the function generate_structure() before calling simulate()"
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
                X, R = datagen.generate()

                corr_test = Eval_Evec_test(X, self.param['Pfa_eval'], self.param['Pfa_evec'],
                                           self.param['bootstrap_count'])
                corr_est, d_cap, u_struc = corr_test.find_structure()
                print(d_cap)
                perf = Metrics(self.corr_truth, corr_est)
                pr, re = perf.PrecisionRecall()
                precision += pr
                recall += re

            prec_vec.append(precision / self.param['num_iter'])
            rec_vec.append(recall / self.param['num_iter'])

        plt.ion()
        if len(self.param['SNR_vec']) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle('Precion recall plots for various SNR')
            ax1.plot(self.param['SNR_vec'], prec_vec, 'o-')
            ax1.set_ylabel('Precision')
            ax1.set_xlabel('SNR')

            ax2.plot(self.param['SNR_vec'], rec_vec, 'o-')
            ax2.set_ylabel('Recall')
            ax2.set_xlabel('SNR')
            plt.show()
        viz = visualization(self.corr_truth, u_struc, self.x_corrs, self.signum, self.n_sets, label_edge=False)
        viz.visualize("True Structure")
        plt.ioff()
        viz_op = visualization(corr_est, u_struc, self.x_corrs, self.signum, self.n_sets, label_edge=False)
        viz_op.visualize("Estimated_structure")


        return corr_est, d_cap

    def run_realData(self, data, disp_struc=True):
        """
        Args:
            data (): must be in the form of a list of ndarrays. Dimensions must be consistent with n_sets and signum
        """

        # assert not self.synthetic_structure and self.n_sets == len(data), "mismatch in input provided"

        # evaluate using Evec and Eval tests
        if 'Pfa_eval' not in self.param:
            self.param['Pfa_eval'] = 0.05
        if 'Pfa_evec' not in self.param:
            self.param['Pfa_evec'] = 0.05
        if 'bootstrap_count' not in self.param:
            self.param['bootstrap_count'] = 1000
        if 'threshold' not in self.param:
            self.param['threshold'] = 0

        #data = np.real(data)
        corr_test = Eval_Evec_test(data, self.param['Pfa_eval'], self.param['Pfa_evec'],
                                   self.param['bootstrap_count'], self.param['threshold'])
        corr_est, d_cap, u_struc = corr_test.find_structure()

        if disp_struc:
            viz_op = visualization(corr_est, u_struc, self.x_corrs, self.signum, self.n_sets, label_edge=False)
            viz_op.visualize("Estimated_structure")
        return corr_est, u_struc, d_cap

    def run(self, *argv, **kwargs):
        if self.simulation_data_type == 'real':
            return self.run_realData(argv[0], **kwargs)

        if self.simulation_data_type == 'synthetic':
            return  self.run_syntheticData()
