"""SVCCA, PWCCA, RSA, and CKA representational similarity metrics."""
import numpy as np
from scipy.stats import spearmanr

def svcca(H0, H1, var_thresh=0.99):
    """SVCCA similarity between two feature matrices.

    H0, H1: n × p matrices (samples × features).
    Returns mean canonical correlation of top SVCCA components.
    """
    H0 = H0 - H0.mean(axis=0, keepdims=True)
    H1 = H1 - H1.mean(axis=0, keepdims=True)

    U0, s0, _ = np.linalg.svd(H0, full_matrices=False)
    U1, s1, _ = np.linalg.svd(H1, full_matrices=False)

    cum0 = np.cumsum(s0**2) / np.sum(s0**2)
    cum1 = np.cumsum(s1**2) / np.sum(s1**2)
    k0 = int(np.searchsorted(cum0, var_thresh) + 1)
    k1 = int(np.searchsorted(cum1, var_thresh) + 1)
    k = min(k0, k1, H0.shape[1], H1.shape[1])

    if k <= 1:
        return 0.0

    _, s_cca, _ = np.linalg.svd(U0[:, :k].T @ U1[:, :k])
    return float(np.mean(s_cca))


def pwcca(H0, H1, y, var_thresh=0.99):
    """PWCCA: SVCCA weighted by target alignment.

    Proper implementation: weights each canonical correlation by how much
    the corresponding canonical direction aligns with the target y
    (Cortes et al., 2012).
    """
    H0 = H0 - H0.mean(axis=0, keepdims=True)
    H1 = H1 - H1.mean(axis=0, keepdims=True)

    U0, s0, _ = np.linalg.svd(H0, full_matrices=False)
    U1, s1, _ = np.linalg.svd(H1, full_matrices=False)

    cum0 = np.cumsum(s0**2) / np.sum(s0**2)
    cum1 = np.cumsum(s1**2) / np.sum(s1**2)
    k0 = int(np.searchsorted(cum0, var_thresh) + 1)
    k1 = int(np.searchsorted(cum1, var_thresh) + 1)
    k = min(k0, k1, H0.shape[1], H1.shape[1])

    if k <= 1:
        return 0.0

    # CCA: SVD of cross-covariance between truncated SV spaces
    U_cca, s_cca, Vt_cca = np.linalg.svd(U0[:, :k].T @ U1[:, :k])

    # Canonical directions in H0 space
    canon_dirs = U0[:, :k] @ U_cca  # n × k

    # Weight = absolute projection of centered target on each canonical direction
    yc = y - y.mean()
    weights = np.abs(canon_dirs.T @ yc)
    weights = weights / (np.sum(weights) + 1e-12)

    return float(np.sum(weights * s_cca))


def rsa(H0, H1):
    """RSA (Representational Similarity Analysis) between two feature matrices.

    Computes Spearman correlation between upper-triangular entries of
    pairwise Euclidean distance matrices.
    """
    n = H0.shape[0]
    D0 = np.zeros((n, n))
    D1 = np.zeros((n, n))
    for i in range(n):
        diff0 = H0[i] - H0
        diff1 = H1[i] - H1
        D0[i] = np.sqrt(np.sum(diff0**2, axis=1))
        D1[i] = np.sqrt(np.sum(diff1**2, axis=1))
    triu_idx = np.triu_indices(n, k=1)
    v0 = D0[triu_idx]
    v1 = D1[triu_idx]
    rho, _ = spearmanr(v0, v1)
    return float(rho)


def cka_from_features(H0, H1):
    """CKA between two feature matrices using linear kernel."""
    K0 = H0 @ H0.T
    K1 = H1 @ H1.T
    n = K0.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K0c = H @ K0 @ H
    K1c = H @ K1 @ H
    num = np.sum(K0c * K1c)
    den = np.sqrt(np.sum(K0c**2) * np.sum(K1c**2))
    return float(num / (den + 1e-12))
