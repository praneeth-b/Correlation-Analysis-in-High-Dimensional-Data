import numpy as np
from itertools import combinations
import random
import math

import scipy as sp
import scipy.linalg as spl

from multipleDatasets.utils.helper import ismember, comb, list_find


class CorrelationStructureGen:
    """
    Description:
    This function generates a correlation structure and augmented covariance matrix for multiple data sets. Correlation
    coefficients for each signal are randomly selected from a normal distribution with a user prescribed mean and
    standard deviation. This function enforces the 'transitive correlation condition.'

    Use the generate() function to create the desired correlation structure.

    """

    def __init__(self, n_sets, tot_corr, corr_means, corr_std, signum, sigmad, sigmaf, maxIters=99):
        """

        Args:
            n_sets (int): Number of datasets
            tot_corr (int): Number of signals/features of the datasets correlated across all datasets
            corr_means (array):  The ith element is the mean of the correlation coefficients associated
                                 with the ith correlated signal.
            corr_std (array): The ith element of the standard deviation of the correlation coefficients associated with
                              the ith correlated signal.
            signum (int): Total number of signals in the dataset
            sigmad (float): Variance of the correlated components
            sigmaf (float): Variance of the independent components
            maxIters (int): Number of random draws of correlation coefficients allowed to find a positive definite
                            correlation matrix.
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
            0], "error('\n Dimension mismatch in corrnum, corr_means and corr_std)"

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
        # assert min(Ev) > 0, "negative eigen value !!! "

        return self.R

    def generate(self):
        """
        Generates the correlation structure of the multi dataset.
        Returns:
            p (ndarray): Matrix of size 'n_sets choose two' x signum. Rows have the same order as x_corrs.
            The ith element of the jth row is the correlation coefficient between the ith signals in the data sets
            indicated by the jth row of self.x_corrs.

            sigma_signals (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block
            correlation matrix of all the data sets. Each block is of size signum x signum and the i-jth block is the
            correlation matrix between data set i and data set j.

            R (ndarray): Matrix of size (n_sets x signum) x (n_sets x signum). Augmented block correlation matrix of
            all the data sets. Each block is of size signum x signum and the i-jth block is the correlation matrix
            between data set i and data set j.

        """

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
                sigma_signals[:, self.corrnum: self.signum] = self.sigmaf * np.ones(
                    (self.n_combi, self.signum - self.corrnum))
            # minEig = 1

            R = self.generateBlockCorrelationMatrix(sigma_signals, p)

            attempts += 1
            e, ev = np.linalg.eig(self.R)
            minEig = np.min(e)
            if attempts > self.maxIters and minEig < 0:
                raise Exception("A positive definite correlation matrix could not be found with prescribed correlation "
                                "structure. Try providing a different correlation structure or reducing the standard "
                                "deviation")

        return p, sigma_signals, R


