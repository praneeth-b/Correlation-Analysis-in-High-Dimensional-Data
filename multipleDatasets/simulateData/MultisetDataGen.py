import numpy as np
from itertools import combinations
import random
import math
import scipy as sp
import scipy.linalg as spl

from multipleDatasets.utils.helper import ismember, comb, list_find


class MultisetDataGen_CorrMeans(object):
    """
    Generate Multiple datasets with prescribed correlation structure
    """

    def __init__(self, subspace_dims, signum, x_corrs, mixing, sigmad, sigmaf, sigmaN, color, n_sets, p, sigma_signals,
                 M, MAcoeff,
                 ARcoeff, Distr, R=0):
        """

        Args:
            subspace_dims ():
            signum ():
            x_corrs ():
            mixing ():
            sigmaN ():
            color ():
            n_sets ():
            p ():
            sigma_signals ():
            M ():
            MAcoeff ():
            ARcoeff ():
            Distr ():
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
        self.sigmad = sigmad
        self.sigmaf = sigmaf
        self.R = R
        self.A = [0] * self.n_sets
        self.S = [0] * self.n_sets
        self.N = [0] * self.n_sets
        self.X = [0] * self.n_sets

    def generateMixingMatrix(self):
        """
        computes the mixing matrices
        Returns:

        """
        # print(self.mixing)
        if self.mixing == 'orth':
            for i in range(self.n_sets):
                orth_Q = spl.orth(np.random.randn(self.subspace_dims[i],
                                                  self.signum))  # np.linalg.qr(np.random.randn(self.subspace_dims[i], self.signum))
                self.A[i] = orth_Q

        elif self.mixing == 'randn':
            for i in range(self.n_sets):
                self.A[i] = np.random.randn(self.subspace_dims[i], self.signum)

        else:
            raise Exception("Unknown mixing matrix property")

    def generateBlockCorrelationMatrix(self):
        """
        Compute the pairwise correlation and assemble the correlation matrices into augmented block correlation matrix
        Returns:

        """
        Rxy = [0] * comb(self.n_sets, 2)
        for i in range(len(self.x_corrs)):
            Rxy[i] = np.sqrt(np.diag(self.sigma_signals[i, :]) * np.diag(self.sigma_signals[i, :])) * np.diag(
                self.p[i, :])  # Assemble correlation matrices into augmented block correlation matrix
        for i in range(self.n_sets):
            t = np.zeros(len(self.x_corrs))
            idx = list_find(self.x_corrs, i)
            t[idx] = 1
            temp = self.sigma_signals[idx, :] == self.sigmad
            temp = temp.max(0)
            self.R[i * self.signum: (i + 1) * self.signum, i * self.signum: (i + 1) * self.signum] = np.diag(
                temp * self.sigmad + np.logical_not(temp) * self.sigmaf)  # recheck the indices

            for j in range(i + 1, self.n_sets):  # check this again
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
        assert min(Ev) > 0, "negative eigen value !!! "
        print("R ready")

    def generateData(self):
        """

        Returns:

        """
        if self.Distr == 'gaussian':
            evr, evec = np.linalg.eig(self.R)
            evr = np.sort(evr)
            fullS = np.matmul(sp.linalg.sqrtm(self.R), np.random.randn(self.n_sets * self.signum, self.M))

        elif self.Distr == 'laplacian':
            signum_aug = self.n_sets * self.signum
            fullS = np.zeros(signum_aug, self.M)
            for m in range(self.M):
                pass  # figure out how to generate laplacian samples in py


        else:
            raise Exception("Unknown source distribution: {}".format(self.Distr))

        for i in range(self.n_sets):
            self.S[i] = fullS[i * self.signum: (i + 1) * self.signum, :]
            self.N[i] = np.sqrt(self.sigmaN) * np.random.randn(self.subspace_dims[i], self.M)

        # add a return if needed

    def filterNoise(self):
        """
        Filter the noise to be colored if specified

        Returns:

        """

        if self.color == 'white':
            return

        if self.color == 'colored':
            for i in range(self.n_sets):
                self.N[i] = sp.signal.lfilter(self.MAcoeff, self.ARcoeff, self.N[i])  # check for correctness

        else:
            raise Exception("Unknkown noise color option")

    def generateNoiseObservation(self):
        """
        Compute the final observation (signal + noise)
        Args:
        Returns:

        """

        for i in range(self.n_sets):
            self.X[i] = self.A[i] @ self.S[i] + self.N[i]

        return self.X

    def generate(self):
        """

        Returns: X, R , A, S

        """
        self.generateMixingMatrix()
        # if self.R.all() == 0:
        #     self.R = np.zeros((self.n_sets * self.signum, self.n_sets * self.signum))
        #     self.generateBlockCorrelationMatrix()
        self.generateData()
        self.filterNoise()
        self.generateNoiseObservation()

        return self.X, self.R, self.A, self.S
