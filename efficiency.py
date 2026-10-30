"""Laplacian spectra, thermodynamic quantities, and integrated efficiency.

Implements the objective defined in theory.md, with automatic full-range
quadrature or integration over a supplied finite grid. Connectivity checks
from net_analysis.py support input validation. simul_annealing_swap.py uses
the compiled objective and validators; usage_example.py and test_efficiency.py
exercise the public functions. See README.md for setup and usage.
"""

import numpy as np
from numba import njit
from net_analysis import _is_connected_dense_kernel


def _validate_adjacency(W, binary=False):
    """Validate the connected, loop-free undirected graphs used here."""
    if np.iscomplexobj(W):
        raise ValueError("W must contain real weights.")
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
        raise ValueError("W must be square with at least two nodes.")
    if not np.all(np.isfinite(W)) or np.any(W < 0):
        raise ValueError("W must contain finite, non-negative weights.")
    if not np.array_equal(W, W.T):
        raise ValueError("W must be symmetric.")
    if np.any(np.diag(W) != 0):
        raise ValueError("W must have a zero diagonal (no self-loops).")
    if binary and np.any((W != 0) & (W != 1)):
        raise ValueError("Annealing requires a binary (0/1) adjacency matrix.")
    if not _is_connected_dense_kernel(W):
        raise ValueError("W must describe a connected graph.")
    return np.ascontiguousarray(W)


def _validate_tau(tau_vec):
    if tau_vec is None:
        return None
    if np.iscomplexobj(tau_vec):
        raise ValueError("tau_vec must contain real values.")
    tau_vec = np.asarray(tau_vec, dtype=np.float64)
    if (tau_vec.ndim != 1 or tau_vec.size < 2
            or not np.all(np.isfinite(tau_vec)) or np.any(tau_vec < 0)
            or np.any(np.diff(tau_vec) <= 0)):
        raise ValueError("tau_vec must have at least two finite, non-negative, strictly increasing values.")
    return np.ascontiguousarray(tau_vec)


@njit
def _prepare_spectrum(eigenvalues):
    """Remove roundoff in the single stationary mode, retaining positive modes."""
    eigenvalues = np.sort(eigenvalues)
    n = eigenvalues.size
    if n < 2 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("The spectrum must contain at least two finite eigenvalues.")
    tolerance = 64.0 * np.finfo(np.float64).eps * n * eigenvalues[-1]
    if (eigenvalues[-1] <= 0 or abs(eigenvalues[0]) > tolerance
            or eigenvalues[1] <= tolerance):
        raise ValueError("Expected one zero eigenvalue and a numerically resolved positive spectral gap.")
    eigenvalues[0] = 0.0
    return eigenvalues


@njit
def _efficiency_at_tau(eigenvalues, t):
    """Stable eta = tau <lambda> / (log N - log Z), with eta(0) = 1."""
    if t == 0.0:
        return 1.0
    n = eigenvalues.size
    remainder = 0.0
    moment = 0.0
    decrement = 0.0
    for j in range(1, n):
        x = t * eigenvalues[j]
        e = np.exp(-x)
        remainder += e
        if np.isfinite(x):
            moment += x * e
        decrement += np.expm1(-x)
    # expm1 avoids cancellation near zero; log1p retains the late-time tail.
    if t * eigenvalues[-1] <= 1.0:
        log_ratio = -np.log1p(decrement / n)
    else:
        log_ratio = np.log(n) - np.log1p(remainder)
    return moment / ((1.0 + remainder) * log_ratio)

@njit
def _spectral_from_eigs(eigenvalues, tau):
    """
    Numba-friendly computation of spectral thermodynamic quantities.
    The information-theoretic quantities are in nats (not bits).
    Used internally in many functions.

    Parameters
    ----------
    eigenvalues : 1D float64 array
    tau         : 1D float64 array 

    Returns
    -------
    S, F, efficiency, C, Z : 1D float64 arrays
    """
    m = tau.shape[0]
    n = eigenvalues.shape[0]

    S = np.empty(m, dtype=np.float64)
    F = np.empty(m, dtype=np.float64)
    efficiency = np.empty(m, dtype=np.float64)
    C = np.empty(m, dtype=np.float64)
    Z = np.empty(m, dtype=np.float64)

    logN = np.log(n)

    for i in range(m):
        t = tau[i]

        if t == 0.0:
            S[i] = logN
            F[i] = -np.inf
            efficiency[i] = 1.0
            C[i] = 0.0
            Z[i] = n
            continue

        # Partition function Z and moments
        Zi = 0.0
        m1 = 0.0   # sum lambda * exp(-t lambda)
        m2 = 0.0   # sum lambda^2 * exp(-t lambda)

        for j in range(n):
            lam = eigenvalues[j]
            e = np.exp(-t * lam)
            Zi += e
            m1 += lam * e
            m2 += lam * lam * e

        Z[i] = Zi

        # Entropy S and free energy F
        Si = (t / Zi) * m1 + np.log(Zi)
        Fi = -np.log(Zi) / t
        S[i] = Si
        F[i] = Fi

        efficiency[i] = _efficiency_at_tau(eigenvalues, t)

        # Heat capacity C
        # C = (t^2/Z) * ( m2 - (m1^2)/Z )
        C[i] = (t * t / Zi) * (m2 - (m1 * m1) / Zi)

    return S, F, efficiency, C, Z



@njit
def symmetric_laplacian_jit(W):
    """
    Compute the symmetric (random-walk) Laplacian
        L = I - D^{-1/2} W D^{-1/2}
    in a Numba-friendly way.

    Parameters
    ----------
    W : 2D np.ndarray (n, n)
        Adjacency/weight matrix (assumed symmetric, non-negative).

    Returns
    -------
    L : 2D np.ndarray (n, n)
        Symmetric normalized Laplacian.
    """
    n = W.shape[0]

    # degree vector d = sum_j W_ij
    d = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += W[i, j]
        d[i] = s

    # d^{-1/2}, guarding against zeros
    d_inv_sqrt = np.empty(n, dtype=np.float64)
    for i in range(n):
        if d[i] > 0.0:
            d_inv_sqrt[i] = 1.0 / np.sqrt(d[i])
        else:
            d_inv_sqrt[i] = 0.0

    # L = -D^{-1/2} W D^{-1/2}
    L = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        di = d_inv_sqrt[i]
        for j in range(n):
            L[i, j] = -di * W[i, j] * d_inv_sqrt[j]

    # Isolated vertices have zero rows in the normalized Laplacian.
    for i in range(n):
        if d[i] > 0.0:
            L[i, i] += 1.0

    return L


def compute_integrated_efficiency(W, tau_vec=None, L_type="RW"):
    """
    Integrate efficiency over [0, infinity), or over a supplied finite grid.

    Parameters
    ----------
    W : 2D np.ndarray (n, n)
        Adjacency/weight matrix (assumed symmetric, non-negative).
    tau_vec : 1D np.ndarray or None, optional
        Finite, non-negative, strictly increasing integration grid. If None,
        choose a range from the spectral gap and refine Simpson quadrature.
    L_type : str, optional
        Type of Laplacian to use: "Comb" for combinatorial, "RW" for random-walk.
        Defaults to "RW" (the symmetric normalized Laplacian).

    Returns
    -------
    int_efficiency : float
        Integrated efficiency. Automatic quadrature targets relative error
        1e-6; the exponentially small tail is truncated (see theory.md).
    """
    if L_type == "Comb":
        L_flag = 0
    elif L_type == "RW":
        L_flag = 1
    else:
        raise ValueError("L_type must be 'Comb' or 'RW'.")

    W = _validate_adjacency(W)
    tau_vec = _validate_tau(tau_vec)
    int_efficiency = _compute_integrated_efficiency(W, tau_vec, L_flag)
    return int_efficiency


@njit 
def _compute_integrated_efficiency(W, tau_vec, L_flag):
    """
    Numba-jitted helper function to compute integrated efficiency.
    
    Parameters
    ----------
    W : 2D np.ndarray (n, n) of float64
        Adjacency/weight matrix (assumed symmetric, non-negative).
    tau_vec : 1D np.ndarray or None
        Supplied finite grid, or a new automatic range for this graph.
    L_flag : int
        Flag indicating type of Laplacian: 0 for combinatorial, 1 for random-walk.
    """
    # Compute Laplacian of graph
    if L_flag == 0:
        H = np.diag(W.sum(axis=1)) - W
    elif L_flag == 1:
        H = symmetric_laplacian_jit(W)
    else:
        raise ValueError("L_flag must be 0 or 1.")

    # Get integrated efficiency
    eigenvalues = np.linalg.eigvalsh(H)
    return _integrate_from_eigs(_prepare_spectrum(eigenvalues), tau_vec)



@njit
def _log_simpson(eigenvalues, a, b, intervals):
    """Simpson quadrature after tau = exp(s); the Jacobian is tau."""
    h = (b - a) / intervals
    total = 0.0
    for i in range(intervals + 1):
        t = np.exp(a + i * h)
        weight = 2.0
        if i == 0 or i == intervals:
            weight = 1.0
        elif i % 2 == 1:
            weight = 4.0
        total += weight * t * _efficiency_at_tau(eigenvalues, t)
    return total * h / 3.0


@njit
def _integrate_from_eigs(eigenvalues, tau_vec):
    if tau_vec is not None:
        efficiency = np.empty(tau_vec.size, dtype=np.float64)
        for i in range(tau_vec.size):
            efficiency[i] = _efficiency_at_tau(eigenvalues, tau_vec[i])
        return np.trapezoid(efficiency, tau_vec)

    # Near zero, eta = 1 + O(tau * lambda_max). At the upper endpoint,
    # the sum of nonstationary heat-trace terms is at most exp(-30).
    a = np.log(1e-8) - np.log(eigenvalues[-1])
    b = np.log(30.0 + np.log(eigenvalues.size - 1)) - np.log(eigenvalues[1])
    intervals = 128
    previous = _log_simpson(eigenvalues, a, b, intervals)
    while intervals < 16384:
        intervals *= 2
        current = _log_simpson(eigenvalues, a, b, intervals)
        if abs(current - previous) <= 15e-6 * abs(current):
            # Richardson correction and the tiny [0, tau_min] contribution.
            return current + (current - previous) / 15.0 + np.exp(a)
        previous = current
    raise ValueError("Automatic integration did not converge; supply a resolved tau_vec.")


def compute_integrated_efficiency_from_lambdas(eigenvalues, tau_vec=None):
    """
    Compute integrated efficiency from eigenvalues of Laplacian.

    Parameters
    ----------
    eigenvalues : (n,) array
        Real, finite Laplacian eigenvalues, in any order, with exactly one
        zero mode. Near-zero roundoff in that mode is removed. A spectral
        gap below 64 * machine_epsilon * n * lambda_max is rejected.
    tau_vec : (k,) array or None, optional
        Finite integration grid, or automatic integration over [0, infinity).
    Returns
    -------
    float
        Integrated efficiency value.
    """

    if np.iscomplexobj(eigenvalues):
        raise ValueError("eigenvalues must be real.")
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a one-dimensional Laplacian spectrum.")
    return _integrate_from_eigs(_prepare_spectrum(eigenvalues), _validate_tau(tau_vec))
