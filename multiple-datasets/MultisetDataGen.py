import numpy as np
from itertools import combinations
import random
import math
import scipy as sp

from helper import ismember, comb


class MultisetDataGen_CorrMeans(object):
    """
    Generate Multiple datasets with prescribed correlation structure
    """

    def __init__(self, subspace_dims, signum, x_corrs, mixing, sigmaN, color, n_sets, p, sigma_signals, M, MAcoeff,
                 ARcoeff, Distr):
        """

        :param subspace_dims:
        :type subspace_dims:
        :param signum:
        :type signum:
        :param x_corrs:
        :type x_corrs:
        :param mixing:
        :type mixing:
        :param sigmaN:
        :type sigmaN:
        :param color:
        :type color:
        :param n_sets:
        :type n_sets:
        :param p:
        :type p:
        :param sigma_signals:
        :type sigma_signals:
        :param M:
        :type M:
        :param MAcoeff:
        :type MAcoeff:
        :param ARcoeff:
        :type ARcoeff:
        :param Distr:
        :type Distr:
        """

        self.subspace_dims = subspace_dims
        self.signum = signum
        self.x_corrs = x_corrs
        self.mixing = mixing
        self.sigmaN = sigmaN
        self.color = color
        self.n_sets = n_sets
        self.p = p
        self.sigma_signals = sigma_signals
        self.M = M
        self.MAcoeff = MAcoeff
        self.ARcoeff = ARcoeff
        self.Distr = Distr

        self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))
        self.A = np.array([0] * self.n_sets)
        self.S = np.array([0] * self.n_sets)
        self.N = np.array([0] * self.n_sets)

    def generateMixingMatrix(self):
        """
        :return:
        :rtype:
        """
        if self.mixing == 'orth':
            for i in range(self.n_sets):
                orth_Q, orth_R = np.linalg.qr(np.random.randn(self.subspace_dims[i], self.signum))
                self.A[i] = orth_Q

        elif self.mixing == 'randn':
            for i in range(self.n_sets):
                self.A[i] = np.random.randn(self.subspace_dims[i], self.signum)

        else:
            raise Exception("Unknown mixing matrix property: {}".format(self.mixing))

    def generateBlockCorrelationMatrix(self):
        Rxy = [0] * comb(self.n_sets, 2)
        for i in range(len(self.x_corrs)):
            Rxy[i] = np.sqrt(np.diag(self.sigma_signals[i, :]) * np.diag(self.sigma_signals[i:, ])) * np.diag(
                self.p[i, :])

        # Assemble correlation matrices into augmented block correlation matrix
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
                self.R[i * self.signum: (i + 1) * self.signum, j * self.signum: (j + 1) * self.signum] = Rxy[c]
                self.R[j * self.signum: (j + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = Rxy[c]

    def generateData(self):
        """
        generate signal S and noise N
        :return:
        :rtype:
        """

        if self.Distr == 'gaussian':
            fullS = sp.linalg.sqrtm(self.R) * np.random.randn(self.n_sets * self.signum, self.M)

        elif self.Distr == 'laplacian':
            signum_aug = self.n_sets * self.signum
            fullS = np.zeros(signum_aug, self.M)
            for m in range(self.M):
                pass  ## figure out how to generate laplacian samples in py


        else:
            raise Exception("Unknown source distribution: {}".format(self.Distr))


        for i in range(self.n_sets):
            self.S[i] = fullS[i*self.signum : (i+1)*self.signum, :]
            self.N[i] = np.sqrt(self.sigmaN)*np.random.randn(self.subspace_dims[i], self.M)
            