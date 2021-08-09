import numpy as np


class Hypothesis_Test(object):
    def __init__(self):
        pass

    def Eigen_value_test(self, P, m_min, P_fa_eval, E, E_star_matrix, B):
        """
        Algorithm 1 of the paper.
        Args:
            P ():
            m_min ():
            P_fa_eval ():
            E ():
            E_star_matrix ():
            B ():

        Returns: the dimension of the correllated subspace

        """
        Lambda = E - 1
        Lambda_star_matrix = E_star_matrix - 1
        smax = m_min - 1  # assuming all datasets have same dimension
        d_cap = 0
        for s in range(-1, smax):
            T = np.sum(np.square(Lambda[s + 1:s + P]))
            Indicators = 0
            for b in range(B):
                T2_star = np.sum(Lambda_star_matrix[b, s + 1:s + P])
                T2_null = T2_star - T
                if T <= T2_null:
                    Indicators += 1

            p_value = Indicators / B
            if p_value >= P_fa_eval:
                d_cap = s + 1,
                break

            if s == smax - 1:
                d_cap = smax
        if type(d_cap) is tuple:
            d_cap = int(d_cap[0])
        return d_cap

    def Eigen_vector_test(self, P, aug_dim, P_fa_evec, d_cap, U, U_star_matrix, B, thresh=0):
        """
        Algorithm 2 of the paper
        Args:
            thresh ():
            P ():
            aug_dim ():
            P_fa_evec ():
            d_cap ():
            U ():
            U_star_matrix ():
            B ():

        Returns: Estimate of the correlation structure.

        """
        U_struc = np.zeros((d_cap, P))
        for i in range(d_cap):

            for p in range(P):
                present = 1
                if p == 0:
                    dim1 = 0
                else:
                    dim1 = int(aug_dim[p - 1] + 1)

                dim2 = int(aug_dim[p])
                T_0 = np.sum(np.square(U[dim1:dim2, i]))
                T = T_0 - thresh
                Indicator = 0
                for b in range(B):
                    T2_star = np.sum(np.square(U_star_matrix[b, dim1:dim2, i]))
                    T2_null = T2_star - T_0
                    if T <= T2_null:
                        Indicator += 1

                p_value = Indicator / B
                if p_value >= P_fa_evec:
                    present = 0

                U_struc[i, p] = present

        return U_struc
