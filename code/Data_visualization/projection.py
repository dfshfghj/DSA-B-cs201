import numpy as np


def projection(alpha: list[np.ndarray], x: list[np.ndarray]):
    A = np.array(alpha).T
    X = np.array(x).T
    C = np.linalg.inv(A.T @ A) @ A.T @ X
    return list(C.T)

def orthogonalization(normal: np.ndarray, out_dim: int):
    in_dim = len(normal)
    orthonormal_basis = [normal]
    basis_count = 1
    for i in range(in_dim):
        if normal[i] == 0:
            basis_count += 1
            base = np.array([0] * in_dim)
            base[i] = 1
            orthonormal_basis.append(base)
        if basis_count == out_dim:
            return orthonormal_basis

