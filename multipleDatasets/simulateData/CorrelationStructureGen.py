import numpy as np
from itertools import combinations
import random
import math
from multipleDatasets.helper import ismember, comb
import scipy as sp
import scipy.linalg as spl

from multipleDatasets.helper import ismember, comb, list_find


class CorrelationStructureGen:
    """
    init implementation of the correlation structure generation function
    Note: all arrays/matrices to be of the form nparray
    """

    def __init__(self, n_sets, tot_corr, corr_means, corr_std, signum, sigmad, sigmaf, maxIters=99):
        """

        :param n_sets:
        :type n_sets:
        :param tot_corr:
        :type tot_corr:
        :param corr_means:
        :type corr_means:
        :param corr_std:
        :type corr_std:
        :param signum:
        :type signum:
        :param sigmad:
        :type sigmad:
        :param sigmaf:
        :type sigmaf:
        :param maxIters:
        :type maxIters:
        """
        self.corrnum = tot_corr.shape[0]
        self.n_sets = n_sets
        self.tot_corr = tot_corr
        self.x_corrs = list(combinations(range(self.n_sets), 2))
        if self.n_sets < 5:
            self.x_corrs = list(reversed(self.x_corrs))
        self.n_combi = len(self.x_corrs)
        self.corr_means = corr_means
        self.corr_std = corr_std
        self.signum = signum
        self.sigmad = sigmad
        self.sigmaf = sigmaf
        self.maxIters = maxIters
        self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))

        assert self.corrnum == np.shape(self.corr_means)[0] == np.shape(self.corr_std)[
            0], "error('\n\tA mean correlation )"
        # "and standard deviation need to " \
        # "be\n\tspecified for each " \
        # "correlated signal.\n\n\tThe " \
        # "number of elements in " \
        # "corr_means is:\t%g\n\tThe " \
        # "number of elements in corr_std " \
        # "is:\t\t%g\n\tThe value of " \
        # "totcorr is:\t\t\t%g\n\n\tThe " \
        # "value of totcorr is equal to " \
        # "number of elements " \
        # "in\n\tcorr_across plus the " \
        # "value of fullcorr.\n\n\tNo data " \
        # "has been generated.\n'," \
        # "size(corr_means,2)," \
        # "size(corr_std,2),corrnum) "

    def generateBlockCorrelationMatrix(self, sigma_signals, p):
        """
                Compute the pairwise correlation and assemble the correlation matrices into augmented block correlation matrix
                Returns:

                """
        Rxy = [0] * comb(self.n_sets, 2)
        for i in range(len(self.x_corrs)):
            Rxy[i] = np.sqrt(np.diag(sigma_signals[i, :]) * np.diag(sigma_signals[i, :])) * np.diag(
                p[i, :])

        # Assemble correlation matrices into augmented block correlation matrix
        for i in range(self.n_sets):
            t = np.zeros(len(self.x_corrs))
            idx = list_find(self.x_corrs, i)
            t[idx] = 1
            temp = sigma_signals[idx, :] == self.sigmad
            temp = temp.max(0)
            self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
                temp * self.sigmad + np.logical_not(temp) * self.sigmaf)

            for j in range(i + 1, self.n_sets):
                a = np.zeros(len(self.x_corrs))
                b = np.zeros(len(self.x_corrs))
                idxa = list_find(self.x_corrs, i)
                idxb = list_find(self.x_corrs, j)
                a[idxa] = 1
                b[idxb] = 1
                # a = np.sum(self.x_corrs == i, 1)
                # b = np.sum(self.x_corrs == j, 2)
                c = np.nonzero(np.multiply(a, b))
                self.R[i * self.signum: (i + 1) * self.signum, j * self.signum: (j + 1) * self.signum] = Rxy[int(c[0])]
                self.R[j * self.signum: (j + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = Rxy[int(c[0])]
        Ev, Uv = np.linalg.eig(self.R)
        #assert min(Ev) > 0, "negative eigen value !!! "
        print("R ready")
        return self.R

    def generate(self):

        minEig = -1
        attempts = 0

        while minEig <= 0:
            p = np.zeros((self.n_combi, self.signum))
            sigma_signals = np.zeros((self.n_combi, self.signum))

            corr_samples = []
            for j in range(self.corrnum):
                t = random.sample(list(range(self.n_sets)), int(self.tot_corr[j]))
                corr_samples.append(t)
                t1 = ismember(self.x_corrs, t)
                t2 = t1.sum(axis=1) == 2

                temp = self.corr_means[j] + self.corr_std[j] * np.random.randn(t2.sum(), 1)
                corr_arr = [0] * len(t2)
                idx = 0
                for k in range(len(t2)):
                    if t2[k] == 1:
                        p[k, j] = max(min(temp[idx], 1), 0)
                        sigma_signals[k, j] = self.sigmad
                        idx += 1
                    else:
                        p[k, j] = 0
                        sigma_signals[k, j] = self.sigmaf

            if self.corrnum < self.signum:
                sigma_signals[:, self.corrnum: self.signum] = self.sigmaf * np.ones((self.n_combi, self.signum - self.corrnum))
            #minEig = 1

            R = self.generateBlockCorrelationMatrix(sigma_signals, p)

            attempts += 1
            e, ev = np.linalg.eig(self.R)
            minEig = np.min(e)
            if attempts > self.maxIters and minEig < 0:
                raise Exception("A positive definite correlation matrix could not be found with prescribed correlation "
                      "structure. Try providing a different correlation structure or reducing the standard deviation")

        return p, sigma_signals, R

    # def egenerateBlockCorrelationMatrix(self):
    #      """
    #     Compute the pairwise correlation and assemble the correlation matrices into augmented block correlation matrix
    #     Returns:
    #
    #     """
    #     Rxy = [0] * comb(self.n_sets, 2)
    #     for i in range(len(self.x_corrs)):
    #         Rxy[i] = np.sqrt(np.diag(self.sigma_signals[i, :]) * np.diag(self.sigma_signals[i,: ])) * np.diag(
    #             self.p[i, :])
    #
    #     # Assemble correlation matrices into augmented block correlation matrix
    #     for i in range(self.n_sets):
    #         t= np.zeros(len(self.x_corrs))
    #         idx = list_find(self.x_corrs, i)
    #         t[idx] = 1
    #         temp = self.sigma_signals[idx, :] == self.sigmad
    #         temp = temp.max(0)
    #         self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
    #             temp * self.sigmad + np.logical_not(temp) * self.sigmaf)  # recheck the indices
    #
    #         for j in range(i+1 , self.n_sets):  # check this again
    #             a = np.zeros(len(self.x_corrs))
    #             b = np.zeros(len(self.x_corrs))
    #             idxa = list_find(self.x_corrs, i)
    #             idxb = list_find(self.x_corrs, j)
    #             a[idxa] = 1
    #             b[idxb] = 1
    #             # a = np.sum(self.x_corrs == i, 1)
    #             # b = np.sum(self.x_corrs == j, 2)
    #             c = np.nonzero(np.multiply(a, b))
    #             self.R[i * self.signum: (i + 1) * self.signum, j * self.signum: (j + 1) * self.signum] = Rxy[int(c[0])]
    #             self.R[j * self.signum: (j + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = Rxy[int(c[0])]
    #     Ev, Uv = np.linalg.eig(self.R)
    #     assert min(Ev) > 0, "negative eigen value !!! "
    #     print("R ready")
    #     return  self.R

    # def ismember(self, A, B):
    #     ret_list = []
    #     for rows in A:
    #         ret_list.append([np.sum(a in B) for a in rows])
    #     return np.array(ret_list)
    #
    # def comb(self, n, r):
    #     f = math.factorial
    #     return int(f(n) / (f(r) * f(n - r)))
