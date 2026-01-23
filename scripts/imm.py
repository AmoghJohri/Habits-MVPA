"""scripts.imm

IMM utilities for fitting mixtures of Beta distributions to devaluation-ratio
data used in the manuscript. Includes initialization, E/M update steps,
helpers for moment-based Beta estimation, KS-distance utilities, a
cross-validated parametric bootstrap for KS, and threshold-finding tools.

Key functions
- `imm(...)`: run IMM and return fitted `(distributions, pi, W)`
- `imm_E`, `imm_M`: E- and M-step implementations
- `ks_distance`, `ks_parametric_bootstrap_cv`: model fit tests via KS
- `find_all_thresholds`: decision boundaries between components
- `simulate_beta_mixture_general`: draw samples from a fitted mixture

Usage
-----
Import and call from analysis code::

    from scripts.imm import imm, ks_parametric_bootstrap_cv
    distributions, pi, W = imm(data, c=2)

Or run as a script to fit and print example devaluation-ratio results::

    python -m scripts.imm

"""
# Importing libraries
import sys
sys.path.append("..")
import numpy          as np
from   tqdm           import tqdm
from   scipy.stats    import beta
from   scipy.optimize import brentq
from   joblib         import Parallel, delayed
from   typing         import List, Tuple, Optional, Dict, Any, Sequence
# Importing custom libraries
from   scripts.util     import *
from   scripts.subjects import Subject

def mixture_cdf(x: float, distributions: List[Tuple[float, float]], pi: List[float]) -> float:
    """Mixture cumulative distribution function at scalar ``x``.

    Parameters
    - x: point at which to evaluate the CDF
    - distributions: list of ``(alpha, beta)`` tuples for Beta components
    - pi: mixing weights for each component (must sum to 1)

    Returns
    - float: value of the mixture CDF at ``x``
    """
    return float(sum(p * beta.cdf(x, a, b) for (a, b), p in zip(distributions, pi)))

def ks_distance(X: Sequence[float], distributions: List[Tuple[float, float]], pi: List[float]) -> float:
    """Kolmogorov–Smirnov distance between empirical CDF of ``X`` and a
    parametric mixture CDF.

    Parameters
    - X: sequence of observations in [0,1]
    - distributions: list of Beta ``(alpha, beta)`` tuples
    - pi: mixing weights

    Returns
    - float: KS distance
    """
    # Empirical CDF at each sorted observation
    X       = np.sort(np.asarray(X, float))
    n       = len(X)
    ecdf    = np.arange(1, n+1) / n
    # Mixture CDF at the same points
    mcdf    = np.array([mixture_cdf(x, distributions, pi) for x in X])
    return float(np.max(np.abs(ecdf - mcdf)))

def get_beta_distribution(X: List[float], y: float, width: float = 1. / 3) -> Tuple[float, float]:
    """Estimate Beta(a,b) parameters from data near ``y`` using method-of-moments.

    The function selects observations within ``[y-width, y+width]`` and
    computes sample mean/variance to return positive ``(a, b)``.
    """
    # Getting the data points within the specified range
    min_x = y - width
    max_x = y + width
    data  = np.asarray([Xi for Xi in X if min_x <= Xi <= max_x])
    # Checking if there are enough data points
    if len(data) < 2:
        raise ValueError("Not enough data points in the specified range.")
    # Getting the expectation and variance
    mu    = np.mean(data)
    var   = np.var(data)
    # Checking if variance is zero
    if var == 0:
        raise ValueError("Variance is zero, cannot compute beta distribution parameters.")
    # Calculating the precision parameter phi
    phi   = (mu * (1 - mu)) / var - 1.
    # Getting the beta distribution parameters
    a     = float(mu * phi)
    b     = float((1 - mu) * phi)
    # Checking if alpha and beta are positive
    if a <= 0 or b <= 0:
        raise ValueError("alpha and beta parameters must be positive.")
    return a, b

def imm_initialize(X: List[float], c: int, seed: int = 42) -> np.ndarray:
    """Initialize IMM by selecting ``c`` seed points from ``X``.

    Returns a sorted NumPy array of initial centers.
    """
    np.random.seed(seed)
    X = X.copy() # to avoid modifying the original list
    Y = []
    for ii in range(c):
        if ii == 0:
            # Select a random index from the list
            idx = [np.random.randint(0, len(X))]
        else:
            # Select the next point based on distance derived probabilities
            D   = [np.square(np.min(np.abs(np.array(Y) - x))) for x in X]
            idx = np.random.choice(len(X), size=1, p=D / np.sum(D))
        # Append the selected value to Y
        Y.append(X[idx[0]])
        # Remove the index from X
        X = np.delete(X, idx)
    return np.sort(Y)

def imm_E(distributions: List[Tuple[float, float]], X: Sequence[float], pi: Optional[List[float]]) -> np.ndarray:
    """
    E-step for an EM algorithm using a mixture of Beta distributions,
    with special handling for observations at x = 0 and x = 1.

    Parameters:
    - distributions: List of [alpha, beta] parameters for each component
    - X: List of input values in [0, 1]
    - pi: Optional list of prior mixture weights; if None, use uniform weights

    Returns:
    - W: Responsibility matrix (components × data), normalized column-wise
    """

    k = len(distributions)  # number of components
    n = len(X)              # number of data points
    # Default to uniform priors if pi is not provided
    if pi is None:
        pi = [1.0 / k] * k
    X = np.array(X)
    W = np.zeros((k, n))
    # Step 1: Identify special components for x = 0 and x = 1
    # j0: component with min alpha, breaking ties with max beta
    # j1: component with min beta, breaking ties with max alpha
    min_alpha = min(d[0] for d in distributions)
    min_beta  = min(d[1] for d in distributions)
    j0_idx    = max((i for i, d in enumerate(distributions) if d[0] == min_alpha), key=lambda i: distributions[i][1])
    j1_idx    = max((i for i, d in enumerate(distributions) if d[1] == min_beta),  key=lambda i: distributions[i][0])
    # Step 2: Compute unnormalized responsibilities
    for i, (a, b) in enumerate(distributions):
        for j, x in enumerate(X):
            if x == 0.:
                W[i, j] = 1.0 if i == j0_idx else 0.0
            elif x == 1.:
                W[i, j] = 1.0 if i == j1_idx else 0.0
            else:
                W[i, j] = pi[i] * beta.pdf(x, a, b)
    # Step 3: Normalize responsibilities column-wise so that sum_i W[i,j] = 1
    col_sums = np.sum(W, axis=0, keepdims=True)
    W /= np.where(col_sums == 0, 1, col_sums)  # prevent division by zero
    return W

def imm_M(W: Sequence[Sequence[float]], X: Sequence[float]) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    M-step for IMM algorithm: computes new Beta distribution parameters
    using moment-matching from responsibility matrix W and data X.

    Parameters:
    - W: Responsibility matrix (k x n) where W[j][i] = P(component j | x_i)
    - X: Data points (length n), assumed to be in [0, 1]

    Returns:
    - distributions: List of (alpha, beta) parameters for each component
    """
    W    = np.asarray(W)
    X    = np.asarray(X)
    k, n = W.shape
    # Step 1: Update mixture weights (averaged responsibilities)
    pi   = np.mean(W, axis=1)  # shape (k,)
    # Step 2: Moment-matching to get new (alpha, beta) per component
    distributions = []
    for j in range(k):
        w = W[j]
        weight_sum = np.sum(w)
        if weight_sum == 0:
            raise ValueError(f"Component {j} has zero responsibility — cannot update.")
        # Weighted mean
        mu = np.dot(w, X) / (n * pi[j])
        # Weighted variance
        var = np.dot(w, (X - mu)**2) / (n * pi[j])
        # Avoid divide-by-zero or degenerate variance
        if var == 0 or mu in (0, 1):
            raise ValueError(f"Degenerate variance or mean for component {j}: mu={mu}, var={var}")
        # Method-of-moments estimation of Beta parameters
        phi = (mu * (1 - mu)) / var - 1
        alpha = mu * phi
        beta_ = (1 - mu) * phi
        if alpha <= 0 or beta_ <= 0:
            raise ValueError(f"Invalid beta parameters for component {j}: alpha={alpha}, beta={beta_}")
        distributions.append((float(alpha), float(beta_)))
    return distributions, pi

def imm(X: Sequence[float], c: int, max_iterations: int = 1000, seed: int = 42, tol: float = 1e-3) -> Tuple[List[Tuple[float, float]], List[float], np.ndarray]:
    """
    Run the IMM algorithm on the data X with c components.
    Returns the final distributions after convergence.

    Parameters:
    - X: List of input values in [0, 1]
    - c: Number of components
    - pi: Optional list of prior mixture weights; if None, use uniform weights

    Returns:
    - distributions: Final Beta distribution parameters for each component
    """
    # Initialize the algorithm
    Y  = imm_initialize(X, c, seed=seed)
    pi = None
    # Get initial distributions
    distributions = [get_beta_distribution(X, y) for y in Y]
    # Expectation-Maximization loop
    for _ in range(max_iterations):
        W                  = imm_E(distributions, X, pi)         # paper E-step
        new_distributions, new_pi = imm_M(W, X)                  # paper MM-step

        # Relative parameter change criterion (per-parameter stationarity)
        deltas = []
        for (a, b), (na, nb) in zip(distributions, new_distributions):
            deltas.append(abs(na - a) / max(abs(na), abs(a), 1e-12))
            deltas.append(abs(nb - b) / max(abs(nb), abs(b), 1e-12))
        if pi is not None:
            for p_old, p_new in zip(pi, new_pi):
                deltas.append(abs(p_new - p_old) / max(abs(p_new), abs(p_old), 1e-12))

        distributions, pi = new_distributions, list(new_pi)
        if deltas and max(deltas) < tol:
            break

    W = imm_E(distributions, X, pi)
    return distributions, pi, W  # W is handy; no log-likelihood is needed

def _stable_diff(pi_j: float, a_j: float, b_j: float, pi_k: float, a_k: float, b_k: float, x: float) -> float:
    """Numerically-stable difference of weighted Beta PDFs at ``x``.

    Computes pi_j * pdf_j(x) - pi_k * pdf_k(x) using log-space to avoid
    underflow/overflow.
    """
    sj = np.log(pi_j) + beta.logpdf(x, a_j, b_j)
    sk = np.log(pi_k) + beta.logpdf(x, a_k, b_k)
    m  = max(sj, sk)  # log-sum-exp trick
    return float(np.exp(sj - m) - np.exp(sk - m))

def find_all_thresholds(
    distributions: List[Tuple[float, float]],
    pi: List[float],
    grid_points: int = 5000,
    eps: float = 1e-6,
    dedup_tol: float = 1e-6
) -> List[float]:
    """
    Return all x in (0,1) where the argmax over components of pi_j * BetaPDF_j(x)
    changes (i.e., decision boundaries between responsibilities).
    """
    K = len(distributions)
    pi = np.asarray(pi, float)
    assert K == len(pi)
    assert np.isclose(pi.sum(), 1.0), "pi must sum to 1"

    # 1) coarse scan with log-responsibilities (numerically stable)
    xg = np.linspace(eps, 1 - eps, grid_points)
    log_resp = np.zeros((K, xg.size))
    for j, (a, b) in enumerate(distributions):
        log_resp[j] = np.log(pi[j]) + beta.logpdf(xg, a, b)

    winners = np.argmax(log_resp, axis=0)

    # 2) find change points in the winner
    change_idxs = np.where(winners[1:] != winners[:-1])[0]

    roots = []
    # 3) refine each boundary with brentq between the two competing components
    for idx in change_idxs:
        j = winners[idx]
        k = winners[idx+1]
        # ensure (j,k) are the two neighbors across the boundary
        # bracket
        xa, xb = xg[idx], xg[idx+1]

        def g(z):
            (a_j, b_j) = distributions[j]
            (a_k, b_k) = distributions[k]
            return _stable_diff(pi[j], a_j, b_j, pi[k], a_k, b_k, z)

        # If the coarse labels differ, g should cross zero in (xa, xb).
        # Still, guard against flat/degenerate cases:
        try:
            fa, fb = g(xa), g(xb)
            if np.sign(fa) == np.sign(fb):
                # expand a tiny bit if needed
                pad = 1e-4
                xa2 = max(eps, xa - pad)
                xb2 = min(1 - eps, xb + pad)
                fa, fb = g(xa2), g(xb2)
                if np.sign(fa) == np.sign(fb):
                    continue
                xa, xb = xa2, xb2
            root = brentq(g, xa, xb, maxiter=200, xtol=1e-12, rtol=1e-10)
            roots.append(root)
        except Exception:
            continue

    # 4) deduplicate very-close roots and sort
    roots = np.array(sorted(roots))
    if roots.size:
        deduped = [roots[0]]
        for r in roots[1:]:
            if abs(r - deduped[-1]) > dedup_tol:
                deduped.append(r)
        roots = deduped
    return list(roots)

def simulate_beta_mixture_general(n_samples: int, distributions: List[Tuple[float, float]], pi: List[float], seed: int = 42) -> np.ndarray:
    """
    Simulate samples from a mixture of Beta distributions.

    Parameters:
    - n_samples: int, number of samples to generate
    - beta_params: list of (alpha, beta) tuples, one for each component
    - pi: list of mixing weights (must sum to 1, same length as beta_params)

    Returns:
    - samples: np.array of shape (n_samples,)
    """
    np.random.seed(seed)   # Setting the seed
    k = len(distributions) # Number of components
    # Check if distributions and pi are valid
    if len(pi) != k:
        raise ValueError("pi and beta_params must be the same length")
    # Check if pi sums to 1
    if not np.isclose(sum(pi), 1.0):
        raise ValueError("Mixing proportions pi must sum to 1")
    # Sample component indices for each point based on mixing proportions
    component_indices = np.random.choice(k, size=n_samples, p=pi)
    # Preallocate samples
    samples = np.empty(n_samples)
    # Sample from each component
    for i in range(k):
        idx = component_indices == i
        n_i = np.sum(idx)
        if n_i > 0:
            alpha_i, beta_i = distributions[i]
            samples[idx] = beta.rvs(alpha_i, beta_i, size=n_i)
    return samples

def _spawn_uint32_seeds(n: int, base_seed: int = 0) -> List[int]:
    """
    Return a list of n distinct uint32 seeds valid for legacy NumPy RNG.
    """
    ss = np.random.SeedSequence(int(base_seed))
    children = ss.spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

def ks_parametric_bootstrap_cv(
    x: Sequence[float],
    c: int = 2,
    B: int = 500,
    Kfolds: int = 4,
    seed: int = 0,
    n_jobs: int = -1,
    return_fold_stats: bool = False,
) -> Dict[str, Any]:
    """
    Parallel cross-validated parametric bootstrap for KS.
    For each bootstrap replicate b:
      - simulate train from the train-fit model, refit on that train,
      - simulate a test set from the refit model,
      - compute KS on the test set vs the refit model,
    and average over folds; compare observed fold-mean KS to the bootstrap
    distribution of fold-mean KS.
    """
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, float)
    n = len(x)

    # --- Build K folds
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, Kfolds)

    # --- Fit once per fold on the true train split; compute observed KS on held-out test
    ks_obs_per_fold = []
    train_models = []  # store (d_tr, p_tr, n_tr, n_te) for each fold
    for f, te_idx in enumerate(folds):
        tr_mask = np.ones(n, dtype=bool); tr_mask[te_idx] = False
        tr_idx = np.where(tr_mask)[0]
        xtr, xte = x[tr_idx], x[te_idx]

        # fit on train
        d_tr, p_tr, _ = imm(xtr, c=c, seed=int(rng.integers(0, 2**32 - 1)))
        train_models.append((d_tr, p_tr, len(xtr), len(xte)))

        # observed KS on test
        ks_obs_per_fold.append(ks_distance(xte, d_tr, p_tr))

    ks_obs_per_fold = np.asarray(ks_obs_per_fold, float)
    ks_obs_mean = float(np.mean(ks_obs_per_fold))

    # --- Pre-generate seeds per replicate (shared across folds for replicate-level aggregation)
    seeds_sim_tr = _spawn_uint32_seeds(B, base_seed=seed + 10_001)
    seeds_fit    = _spawn_uint32_seeds(B, base_seed=seed + 20_001)
    seeds_sim_te = _spawn_uint32_seeds(B, base_seed=seed + 30_001)

    # Helper to run one bootstrap replicate across all folds and average
    def _one_rep(rep_idx):
        rng_rep = np.random.default_rng(int(seeds_fit[rep_idx]))
        ks_vals = []
        for f, (d_tr, p_tr, n_tr, n_te) in enumerate(train_models):
            # simulate train from train-fit model
            xb_tr = simulate_beta_mixture_general(n_tr, d_tr, p_tr, seed=seeds_sim_tr[rep_idx] + f)
            # refit on bootstrap train
            db_tr, pb_tr, _ = imm(xb_tr, c=c, seed=int(rng_rep.integers(0, 2**32 - 1)))
            # simulate test from refit model
            xb_te = simulate_beta_mixture_general(n_te, db_tr, pb_tr, seed=seeds_sim_te[rep_idx] + f)
            # KS on bootstrap test vs refit model
            ks_vals.append(ks_distance(xb_te, db_tr, pb_tr))
        return float(np.mean(ks_vals))

    ks_null_means = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_one_rep)(b) for b in range(B)
    )
    ks_null_means = np.asarray(ks_null_means, float)

    p = (np.sum(ks_null_means >= ks_obs_mean) + 1.0) / (B + 1.0)

    out = {
        "KS_obs_per_fold": ks_obs_per_fold.tolist(),
        "KS_obs_mean": ks_obs_mean,
        "p_value": float(p),
    }
    if return_fold_stats:
        out["KS_null_means"] = ks_null_means
    return out

def main():
    # Loading the subjects
    subjects              = Subject.get_subjects()
    subids                = Subject.get_subids(subjects)
    # Loading the devaluation ratios
    devaluation_ratios    = BEHAVIOR[BEHAVIOR['subID'].isin(subids)]['devaluation_ratio'].tolist()
    devaluation_ratios    = np.sort(np.asarray(devaluation_ratios))
    # Fitting the beta-model
    component             = 2
    seed                  = 42
    distributions, pi, W  = imm(devaluation_ratios, component, seed=seed)
    ks                    = ks_distance(devaluation_ratios, distributions, pi)
    # Round everything off to the first 3 decimal places for display
    distributions         = [(round(a, 3), round(b, 3)) for (a, b) in distributions]
    pi                    = [float(round(p, 3)) for p in pi]
    # Find all thresholds
    thresholds            = find_all_thresholds(distributions, pi)
    thresholds            = [float(round(t, 3)) for t in thresholds]
    # Printing the results 
    print("Distributions:", distributions)
    print("Mixing proportions:", pi)
    print("Thresholds:", thresholds)
    print("KS distance:", round(ks, 3))

if __name__ == "__main__":
    main()