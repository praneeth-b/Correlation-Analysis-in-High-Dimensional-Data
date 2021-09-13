import numpy as np


class Hypothesis_Test(object):
    """
    Description: This class implemts the hypothesis tests for the estimation of the number of correlated features
    accross multiple datasets and their structure.
    Composed of 2 hypothesis tests:

    1) Eigen value test: Hypothesis test to detect and compute the number of eigen values of the
       coherence matrix that are greater than '1'. This determines the dimesion of the correlated
       subspace 'd'

    2) Eigen vector test: Hypothesis test to detect if the eigen vectors corresponding to the 'd' largest
       eigen values are '0' or '1'. This determines the correlated signals across the datasets.

    Reference: T. Hasija, C. Lameiro, T. Marrinan and P. J. Schreier,"Determining the Dimension and
               Structure of the Subspace Correlated Across Multiple Data Sets."

    """

    def __init__(self):
        pass

    def Eigen_value_test(self, P, m_min, P_fa_eval, E, E_star_matrix, B):
        """
        Algorithm 1 of the paper in 'Reference'
        Args:
            P (int): num of datasets
            m_min (int): number of signals in each dataset
            P_fa_eval (float): False alarm probability
            E (ndarray): eigen values of the coherence matrix of the original data
            E_star_matrix (ndarray): concatenated eigen values of the coherence matrix corresponding to the
                                     corresponding bootstrapped samples
            B (int): number of bootstraps done

        Returns: the dimension of the correllated subspace 'd'

        """
        Lambda = E - 1
        Lambda_star_matrix = E_star_matrix - 1
        smax = m_min # assuming all datasets have same dimension
        d_cap = 0
        for s in range(smax):
            T = np.sum(np.square(Lambda[s:s + P]))
            Indicators = 0
            for b in range(B):
                T2_star = np.sum(np.square(Lambda_star_matrix[s:s + P, b]))
                T2_null = T2_star - T
                if T <= T2_null:
                    Indicators += 1

            p_value = Indicators / B
            if p_value >= P_fa_eval:
                d_cap = s
                break

            if s == smax - 1:
                d_cap = smax
        if type(d_cap) is tuple:
            d_cap = int(d_cap[0])
        return d_cap

    def Eigen_vector_test(self, P, aug_dim, P_fa_evec, d_cap, U, U_star_matrix, B, thresh=0):
        """
        Algorithm 2 of the paper in 'Reference'
        Args:
            thresh (float): Threshold for the hypothesis test
            P (int): number of datasets
            aug_dim (array):
            P_fa_evec (float): False alarm probability
            d_cap (int): estimated dimension of the correlated subspace
            U (ndarray): Eigen vectors of the coherence matrix correponding to the original dataset
            U_star_matrix (ndarray): concatenated eigen vectors of the coherence matrix corresponing to the
                                     bootstrapped dataset.
            B (int): number of bootstraps done.

        Returns: Estimate of the correlation structure.

        """
        U_struc = np.zeros((d_cap, P))
        for i in range(d_cap):

            for p in range(P):
                present = 1
                if p == 0:
                    dim1 = 0
                else:
                    dim1 = int(aug_dim[p - 1] )

                dim2 = int(aug_dim[p])
                T_0 = np.sum(np.square(U[dim1:dim2, i]))


                T = T_0 - thresh
                Indicator = 0
                for b in range(B):
                    T2_star = np.sum(np.square(U_star_matrix[b][dim1:dim2, i]))
                    T2_null = T2_star - T_0
                    if T <= T2_null:
                        Indicator += 1

                p_value = Indicator / B
                if p_value >= P_fa_evec:
                    present = 0

                U_struc[i, p] = present

        return U_struc
