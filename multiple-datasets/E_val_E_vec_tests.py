import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, block_diag
import scipy as sp
from helper import arr_sort

class Eval_Evec_test(object):
    def __init__(self, X_cell, P_fa_eval, P_fa_evec, B):
        """

        Args:
            X_cell (nparray): list of dataset ndarrays
            P_fa_eval ():
            P_fa_evec ():
            B ():
        """
        self.x_cell = X_cell
        self.P_fa_eval = P_fa_eval
        self.P_fa_evec = P_fa_evec
        self.B = B

    def augmentData(self, x_cell):
        """
        Returns: augmented data and covariance matrices of individual data sets and dimension of each dataset

        """
        P = self.x_cell.shape[0]  # number of datasets
        M = self.x_cell[0].shape[1]  # number of samples
        x_aug = []
        m = []  # dimension of each dataset
        Rxx_mH = np.array([0] * P)
        for i in range(P):
            x_aug.extend(x_cell)
            Rxx_mH[i] = sqrtm(inv(np.matmul(x_cell[i], np.transpose(x_cell[i])) / M))
            m.append(x_cell[i].shape[0])
        return x_aug, Rxx_mH, m

    def generateInv_RD_cap(self):
        """
          Returns: Rd square root(inverse of RD matrix)

        """
        P = self.x_cell.shape[0]  # number of datasets
        M = self.x_cell[0].shape[1]  # number of samples

        _, Rxx_mH, m = self.augmentData(self.x_cell)
        aug_dim = np.array([0]*P)
        Rd_mh = Rxx_mH[0]
        aug_dim[0] = m[0]
        for i in range(1,P):
            Rd_mh = block_diag(Rd_mh,Rxx_mH[i])
            aug_dim[i] = aug_dim[i-1]+m[i]

        return Rd_mh, aug_dim

    def generateR_xx_aug(self, data_cell):
        """
        Returns:  augmented data covariance matrix

        """
        X_aug, _, _ = self.augmentData(data_cell)
        Rxx_aug = np.matmul(X_aug, X_aug.T)
        return Rxx_aug

    def generate_C(self, data_cell):
        """
        Returns: Augmented coherence matrix

        """
        R_cap = self.generateR_xx_aug(data_cell)
        R_d = self.generateInv_RD_cap()
        self.C = np.matmul(R_d, np.matmul(R_cap, R_d))
        return self.C


    def calc_Eval_Evec(self, mat):
        """

        Args:
            mat (np.array): input square matrix

        Returns: eigen value and eigen vector sorted by absolute values of eigen values

        """
        E, U = np.linalg.eig(mat)
        E = np.array(list(map(abs, E)))  # absolute value, if complex eigen values
        E, idx = arr_sort(E, order='descending')
        U = U[:, idx]
        return E, U

    def bootstrap(self, data_cell, num_samples):
        """
        bootstraps samples out of data_cell and returns bootstrapped samples
        Args:
            num_samples (int):
            data_cell (cell): list of ndarrays for

        Returns: data cell i.e. list of datasets for different modalities bootstrapped from original dataset

        """
        P = data_cell.shape[0]
        bs_cell = np.array([0]*P)
        M = data_cell[0].shape[1]  # number of samples
        idx_list = list(range(M))
        bs_idx = np.random.choice(idx_list, replace=True, size=M)

        for i in range(P):
            bs_cell[i] = data_cell[:, bs_idx]

        return bs_cell



    def main_algo(self):





























