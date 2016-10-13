""" 
Utility functions, used by different methods
"""

def _svi_lower_triang_mat_to_vec(mat):
        """
        Transforms a lower-triangular matrix to a vector of it's components, that are lower than the main diagonal
        :param mat: lower-triangular matrix
        :return: a vector
        """
        indices = np.tril_indices(mat.shape[0])
        vec = mat[indices]
        return vec

def _svi_lower_triang_vec_to_mat(vec):
        """
        Transforms a vector similar to the ones, produced by _svi_lower_triang_mat_to_vec, to a lower-diagonal matrix
        :param vec: a vector of the lower-triangular matrix' components, that are lower than the main diagonal
        :return: a lower-triangular matrix
        """
        m = len(vec)
        k = (-1 + np.sqrt(1 + 8 * m)) / 2
        if k != int(k):
            raise ValueError("Vec has an invalid size")
        indices = np.tril_indices(k)
        mat = np.zeros((int(k), int(k)))
        mat[indices] = vec.reshape(-1)
        return mat

def _get_inv_logdet_cholesky(mat):
        try:
            L = np.linalg.cholesky(mat)
        except:
            L = np.linalg.cholesky(_eig_val_correction(mat, eps=1e-1))
        L_inv = np.linalg.inv(L)
        mat_inv = L_inv.T.dot(L_inv)
        mat_logdet = np.sum(np.log(np.diag(L))) * 2
        return mat_inv, mat_logdet