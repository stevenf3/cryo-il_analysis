import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================
# USER SETTINGS
# ==============================
FILE_PATH = "GaN_68K_200ms_Start.csv"  # <-- change this
X_COLUMN = "Wavelength_nm"                 # or "Wavelength_nm"
Y_COLUMN = "Counts"

# ==============================
# LOADING AND PRE-PROCESSING
# ==============================
df = pd.read_csv(FILE_PATH)

if X_COLUMN not in df.columns or Y_COLUMN not in df.columns:
    raise ValueError(f"Columns {X_COLUMN!r} and {Y_COLUMN!r} must exist in the file.")

x = df[X_COLUMN].to_numpy(dtype=float)
y_raw = df[Y_COLUMN].to_numpy(dtype=float)

# Sort by x just in case
order = np.argsort(x)
x = x[order]
y_raw = y_raw[order]

# Simple baseline shift + normalization for fitting stability
y = y_raw - np.min(y_raw)
if np.max(y) > 0:
    y = y / np.max(y)

# ==============================
# HU-STYLE GAUSSIAN BASIS FUNCTIONS
# ==============================
def forward_model(A, sigma, n_points):
    """
    Discrete Hu-type model:
    f(j) = sum_i A_i * exp(-(j-i)^2 / (2 sigma_i^2))
    Using indices j,i = 0..N-1 (Python indexing)
    """
    idx = np.arange(n_points)
    j = idx[:, None]      # shape (N,1)
    i = idx[None, :]      # shape (1,N)

    sigma_safe = sigma[None, :] + 1e-12
    exponent = -0.5 * ((j - i) ** 2) / (sigma_safe ** 2)
    G = np.exp(exponent)  # (N,N)

    return G @ A          # (N,)


def compute_error_and_grads(y, A, sigma):
    """
    Compute E, dE/dA, dE/dsigma for Hu-type model.
    """
    n_points = len(y)
    f = forward_model(A, sigma, n_points)
    residual = f - y

    idx = np.arange(n_points)
    j = idx[:, None]
    i = idx[None, :]

    sigma_safe = sigma[None, :] + 1e-12
    exponent = -0.5 * ((j - i) ** 2) / (sigma_safe ** 2)
    G = np.exp(exponent)

    # dE/dA_i = sum_j residual(j) * exp(...)
    dE_dA = (residual[:, None] * G).sum(axis=0)

    # dE/dsigma_i = sum_j residual(j) * exp(...) * A_i * (j-i)^2 / sigma_i^3
    diff2 = (j - i) ** 2
    A_safe = A[None, :]
    sigma3 = sigma_safe ** 3
    dE_dsigma = (residual[:, None] * G * A_safe * diff2 / sigma3).sum(axis=0)

    E = 0.5 * np.sum(residual ** 2)
    return E, dE_dA, dE_dsigma


def estimate_Tmax(y, E_max_factor=1e-4, p_A=0.1, p_sigma=0.1, alpha=0.3, max_iter=2000, rng=None):
    """
    Roughly estimate T_max as in Hu et al.
    """
    n = len(y)
    if rng is None:
        rng = np.random.default_rng(0)

    A = rng.random(n)
    sigma = rng.random(n)

    dA = np.zeros_like(A)
    dsigma = np.zeros_like(sigma)

    E0, _, _ = compute_error_and_grads(y, A, sigma)
    E_max = E0 * E_max_factor

    t = 0
    while t < max_iter:
        E, dE_dA, dE_dsigma = compute_error_and_grads(y, A, sigma)
        if E <= E_max:
            break

        # gradient + inertia
        dA = -p_A * dE_dA + alpha * dA
        dsigma = -p_sigma * dE_dsigma + alpha * dsigma

        A += dA
        sigma += dsigma

        A = np.clip(A, 0, None)
        sigma = np.clip(sigma, 1e-3, n / 2)

        t += 1

    T_max = max(3 * t, 50)  # minimum safety
    return T_max


def extract_peaks_for_r(y, T_max, r, p_A=0.05, p_sigma=0.05, alpha=0.3, k1=2.0, k2=2.0, rng=None):
    """
    Stage 2: for a given MIS r, use Hu's peak extraction to find candidate
    peak positions (indices where A_i > 0 at the end).
    """
    n = len(y)
    if rng is None:
        rng = np.random.default_rng(0)

    # Start from random A and narrow widths
    A = rng.random(n)
    sigma = np.full(n, 2.0)

    dA = np.zeros_like(A)
    dsigma = np.zeros_like(sigma)

    T_max1 = int(0.4 * T_max)
    T_max1 = max(T_max1, 10)

    for t in range(1, T_max1 + 1):
        E, dE_dA, dE_dsigma = compute_error_and_grads(y, A, sigma)

        dA = -p_A * dE_dA + alpha * dA
        dsigma = -p_sigma * dE_dsigma + alpha * dsigma

        A += dA
        sigma += dsigma

        A = np.clip(A, 0, None)
        sigma = np.clip(sigma, 1e-3, n / 2)

        # Peak suppression based on local maxima and time-dependent thresholds
        kk1 = np.exp(-k1 * (1 - t / T_max1))
        kk2 = np.exp(-k2 * (1 - t / T_max1))
        radius = int(max(1, r * kk1))

        A_new = A.copy()
        for i in range(n):
            left = max(0, i - radius)
            right = min(n, i + radius + 1)
            local_max = A[left:right].max()
            if local_max <= 0:
                A_new[i] = 0.0
            else:
                if A[i] < local_max * kk2:
                    A_new[i] = 0.0
        A = A_new

    peak_indices = np.where(A > 0)[0]
    return peak_indices


def cluster_peak_indices(indices, min_gap):
    """
    Collapse nearby indices into clusters, then take the mean as peak center.
    """
    if len(indices) == 0:
        return []
    clusters = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx - current[-1] <= min_gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    centers = [int(np.mean(c)) for c in clusters]
    return centers


# ==============================
# STANDARD GAUSSIAN FITTING
# ==============================
def gaussian_sum(x, *params):
    """
    Sum of Gaussians in physical x-space:
    params = [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    """
    n_peaks = len(params) // 3
    y = np.zeros_like(x)
    for k in range(n_peaks):
        A = params[3 * k]
        mu = params[3 * k + 1]
        sig = params[3 * k + 2]
        y += A * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    return y


def refine_with_curve_fit(x, y, peak_centers_idx, amp_guess_factor=1.0, width_guess=None):
    """
    Use SciPy curve_fit to refine amplitudes, centers, and widths
    based on detected peak centers (indices).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    centers_x = x[peak_centers_idx]
    A0 = y[peak_centers_idx] * amp_guess_factor

    if width_guess is None:
        width_guess = 0.1 * (x.max() - x.min())

    sig0 = np.full_like(A0, width_guess, dtype=float)

    p0 = []
    bounds_lower = []
    bounds_upper = []

    for A_i, mu_i, sig_i in zip(A0, centers_x, sig0):
        p0.extend([A_i, mu_i, sig_i])
        bounds_lower.extend([0.0, x.min(), 0.0])
        bounds_upper.extend([np.inf, x.max(), x.max() - x.min()])

    popt, pcov = curve_fit(
        gaussian_sum, x, y, p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=20000
    )
    return popt, pcov


# ==============================
# WRAPPER: FULL HU-STYLE WORKFLOW
# ==============================
def hu_deconvolve_PL(x, y, r_values=None, rng=None):
    """
    Full Hu-style deconvolution on one PL spectrum.
    Returns:
      best_r, E_r, centers_r, popt, pcov
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)

    if rng is None:
        rng = np.random.default_rng(0)

    if r_values is None:
        # Minimal but reasonable range of MIS values
        r_values = np.arange(1, max(3, n // 20))

    # Stage 1: estimate iteration count T_max
    T_max = estimate_Tmax(y, rng=rng)
    print(f"Estimated T_max = {T_max}")

    E_r = {}
    centers_r = {}

    for r in r_values:
        peak_idx = extract_peaks_for_r(y, T_max, r, rng=rng)
        centers_idx = cluster_peak_indices(peak_idx, min_gap=r)
        centers_r[r] = centers_idx

        if len(centers_idx) == 0:
            E_r[r] = np.inf
            continue

        # Quick approximate error E(r) using a crude Gaussian sum
        width_guess = 0.1 * (x.max() - x.min())
        params0 = []
        for c_idx in centers_idx:
            mu = x[c_idx]
            A0 = y[c_idx]
            params0.extend([A0, mu, width_guess])

        f_guess = gaussian_sum(x, *params0)
        E_r[r] = 0.5 * np.sum((f_guess - y) ** 2)

    # Choose best_r from E(r): look for low error before big jump
    rs = np.array(sorted(E_r.keys()))
    Es = np.array([E_r[rv] for rv in rs])

    dE = np.diff(Es)
    if len(dE) == 0:
        best_r = rs[0]
    else:
        jump_idx = np.argmax(dE)
        if jump_idx > 0:
            si_rs = rs[:jump_idx + 1]
            si_Es = Es[:jump_idx + 1]
        else:
            si_rs = rs
            si_Es = Es
        best_r = si_rs[np.argmin(si_Es)]

    print(f"Selected best_r = {best_r}")

    centers_idx_best = centers_r[best_r]
    popt, pcov = refine_with_curve_fit(x, y, centers_idx_best)

    return best_r, E_r, centers_r, popt, pcov


# ==============================
# RUN DECONVOLUTION
# ==============================
best_r, E_r, centers_r, popt, pcov = hu_deconvolve_PL(x, y)

print("\nFinal fitted peaks (in physical units of", X_COLUMN, "):")
n_peaks = len(popt) // 3
for k in range(n_peaks):
    A = popt[3 * k]
    mu = popt[3 * k + 1]
    sig = popt[3 * k + 2]
    print(f"  Peak {k+1}: A={A:.4g}, center={mu:.6g}, sigma={sig:.4g}")

# ==============================
# PLOTTING
# ==============================
y_fit = gaussian_sum(x, *popt)
residuals = y - y_fit

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Data (normalized)", linewidth=1.5)
plt.plot(x, y_fit, "--", label="Total fit", linewidth=1.5)

# Plot individual Gaussians
for k in range(n_peaks):
    A = popt[3 * k]
    mu = popt[3 * k + 1]
    sig = popt[3 * k + 2]
    comp = A * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    plt.plot(x, comp, ":", label=f"Peak {k+1}")

plt.xlabel(X_COLUMN)
plt.ylabel("Normalized Counts")
plt.legend()
plt.title("Gaussian Deconvolution of PL Spectrum")
plt.tight_layout()
plt.savefig("PL_spectrum_fit.png", dpi=300)


plt.figure(figsize=(8, 3))
plt.plot(x, residuals, label="Residuals")
plt.axhline(0, color="k", linewidth=0.8)
plt.xlabel(X_COLUMN)
plt.ylabel("Residual")
plt.title("Fit Residuals")
plt.tight_layout()
plt.savefig("PL_spectrum_residuals.png", dpi=300)
plt.show()
