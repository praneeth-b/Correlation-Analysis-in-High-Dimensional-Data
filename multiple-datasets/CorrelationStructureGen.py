import numpy as np
from itertools import combinations
import random
import math
from helper import ismember, comb

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
        self.corrnum = tot_corr.shape[1]
        self.n_sets = n_sets
        self.tot_corr = tot_corr
        self.x_corrs = list(combinations(range(1, self.n_sets + 1), 2))
        self.n_combi = self.x_corrs.shape[0]
        self.corr_means = corr_means
        self.corr_std = corr_std
        self.signum = signum
        self.sigmad = sigmad
        self.sigmaf = sigmaf
        # self.maxIters = maxIters
        self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))

        assert self.corrnum == np.shape(self.corr_means) == np.shape(self.corr_std), "error('\n\tA mean correlation " \
                                                                                     "and standard deviation need to " \
                                                                                     "be\n\tspecified for each " \
                                                                                     "correlated signal.\n\n\tThe " \
                                                                                     "number of elements in " \
                                                                                     "corr_means is:\t%g\n\tThe " \
                                                                                     "number of elements in corr_std " \
                                                                                     "is:\t\t%g\n\tThe value of " \
                                                                                     "totcorr is:\t\t\t%g\n\n\tThe " \
                                                                                     "value of totcorr is equal to " \
                                                                                     "number of elements " \
                                                                                     "in\n\tcorr_across plus the " \
                                                                                     "value of fullcorr.\n\n\tNo data " \
                                                                                     "has been generated.\n'," \
                                                                                     "size(corr_means,2)," \
                                                                                     "size(corr_std,2),corrnum) "

    def generate(self, max_iters=99):
        """

        :param max_iters:
        :type max_iters:
        :return:
        :rtype:
        """
        minEig = -1
        attempts = 0

        while minEig <= 0:
            p = np.zeros((self.n_combi, self.signum))
            sigma_signals = np.zeros((self.n_combi, self.signum))

            for j in range(self.corrnum):
                t = random.sample(list(range(1, self.n_sets)), self.tot_corr[j])
                t = ismember(self.x_corrs, t)
                t = t.sum(axis=1) == 2
                temp = self.corr_means(j) + self.corr_std(j) * np.random.randn(t.shape[0], 1)
                p[t, j] = max(min(temp, 1), 0)
                sigma_signals[np.where(t == 1), j] = self.sigmad
                sigma_signals[np.where(t == 0), j] = self.sigmaf

            if self.corrnum < self.signum:
                sigma_signals[:, self.corrnum:self.signum] = self.sigmaf * np.ones(self.n_combi,
                                                                                   self.signum - self.corrnum)

            # compute pairwise correllation matrices
            R_xy = [0] * comb(self.n_sets, 2)
            for i in range(len(self.x_corrs)):
                R_xy[i] = np.sqrt(np.diag(sigma_signals[i, :]) * np.diag(sigma_signals[i:, ])) * np.diag(p[i, :])

            # Assemble correlation matrices into augmented blk of correlation matrix
            for i in range(self.n_sets):
                t = np.sum(self.x_corrs == i, 1)
                temp = sigma_signals[np.nonzero(t), :] == self.sigmad
                temp = temp.max(0)

                self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
                    temp * self.sigmad + np.logical_not(temp) * self.sigmaf)  # recheck the indices

                for j in range(i + 1, self.n_sets):  # check this again
                    a = np.sum(self.x_corrs == 1, 1)
                    b = np.sum(self.x_corrs == j, 2)
                    c = np.nonzero(np.multiply(a, b))
                    self.R[i * self.signum: (i + 1) * self.signum, j * self.signum: (j + 1) * self.signum] = R_xy[c]
                    self.R[j * self.signum: (j + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = R_xy[c]

            attempts += 1
            e, ev = np.linalg.eig(self.R)
            minEig = np.min(e)
            if attempts > max_iters and minEig < 0:
                print("A positive definite correlation matrix could not be found with prescribed correlation "
                      "structure. ")

        return p, sigma_signals, self.R

    # def ismember(self, A, B):
    #     ret_list = []
    #     for rows in A:
    #         ret_list.append([np.sum(a in B) for a in rows])
    #     return np.array(ret_list)
    #
    # def comb(self, n, r):
    #     f = math.factorial
    #     return int(f(n) / (f(r) * f(n - r)))



