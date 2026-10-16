"""
Encoding-model pieces that need only numpy, scipy and scikit-learn.

Everything that maps model features to neural responses lives here so it can
be tested without torch or the Allen SDK.

Conventions
- responses_trials: array (n_images, n_trials, n_neurons), one row per
  presentation of each image. Missing trials are NaN.
- responses: array (n_images, n_neurons), the trial mean.
- features: array (n_images, n_features).

The noise ceiling is per neuron, split-half across trials, Spearman-Brown
corrected. That is the quantity Conwell et al. 2021 and Madan et al. 2024 use.
Splitting across neurons instead (what this repo did before) measures how
consistent the population average is and says nothing about how predictable a
single neuron is.
"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

RIDGE_ALPHAS = np.logspace(-2, 5, 8)


def split_half_reliability(responses_trials: np.ndarray, n_splits: int = 20, seed: int = 0) -> np.ndarray:
    """Per-neuron split-half reliability across trials, Spearman-Brown corrected.

    For each neuron: split the trials of every image into two random halves,
    average each half across trials, correlate the two half-means across
    images, then correct for the halving with r_full = 2r / (1 + r). Repeat
    n_splits times and average. Returns an array (n_neurons,) in [-1, 1].
    """
    n_images, n_trials, n_neurons = responses_trials.shape
    rng = np.random.RandomState(seed)
    out = np.zeros((n_splits, n_neurons))
    for s in range(n_splits):
        perm = rng.permutation(n_trials)
        half_a, half_b = perm[: n_trials // 2], perm[n_trials // 2:]
        mean_a = np.nanmean(responses_trials[:, half_a, :], axis=1)
        mean_b = np.nanmean(responses_trials[:, half_b, :], axis=1)
        for j in range(n_neurons):
            a, b = mean_a[:, j], mean_b[:, j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
                out[s, j] = 0.0
                continue
            r = pearsonr(a[ok], b[ok])[0]
            out[s, j] = 2 * r / (1 + r) if r > -1 else -1.0
    return out.mean(axis=0)


def select_reliable_neurons(reliability: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of neurons whose Spearman-Brown reliability is at least threshold."""
    return np.where(reliability >= threshold)[0]


def reduce_features(features: np.ndarray, n_components: int = 4096, seed: int = 0) -> np.ndarray:
    """Sparse random projection to n_components, as in Conwell et al. 2021.

    Lets whole spatial feature maps (hundreds of thousands of dims) be used
    without PCA. Feature sets already at or below n_components are returned as is.
    """
    if features.shape[1] <= n_components:
        return features.astype(np.float32)
    proj = SparseRandomProjection(n_components=n_components, random_state=seed)
    return proj.fit_transform(features).astype(np.float32)


def cv_ridge_scores(features: np.ndarray, responses: np.ndarray, n_folds: int = 5, seed: int = 0,
                    alphas: np.ndarray = RIDGE_ALPHAS) -> Dict[str, np.ndarray]:
    """Cross-validated ridge regression from features to every neuron at once.

    Outer K-fold over images. Inside each fold: standardise on the training
    images only, then RidgeCV with one alpha per neuron chosen by efficient
    leave-one-out on the training images. Predictions are gathered across folds
    and scored once per neuron on the full held-out vector, which is far more
    stable than averaging per-fold correlations on 20-image folds.

    Returns dict with 'predictions' (n_images, n_neurons) and 'r' (n_neurons,).

    When the features carry no information about a neuron this scorer gives a
    slightly negative r (about -0.1 at 118 images), because each fold's model
    regresses toward its own training mean. A negative median across neurons
    means 'nothing learned', not 'anti-predicts'.
    """
    n_images, n_neurons = responses.shape
    predictions = np.zeros((n_images, n_neurons), dtype=np.float32)
    kf = KFold(n_splits=min(n_folds, n_images), shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(features):
        scaler = StandardScaler().fit(features[train_idx])
        x_train = scaler.transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(x_train, responses[train_idx])
        predictions[test_idx] = model.predict(x_test)
    r = np.array([score_pair(responses[:, j], predictions[:, j]) for j in range(n_neurons)], dtype=np.float32)
    return {"predictions": predictions, "r": r}


def score_pair(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r, or 0 when either side is constant."""
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    r = pearsonr(y_true, y_pred)[0]
    return float(r) if np.isfinite(r) else 0.0


def normalised_r2(r: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Explained variance as a fraction of the per-neuron ceiling.

    r² of the prediction divided by the Spearman-Brown reliability of the
    neuron (the variance a perfect model of the trial mean could explain).
    Only defined where the reliability is positive. Not clipped at 1: a value
    above 1 is a sign the ceiling is underestimated and should be visible.
    """
    out = np.full_like(r, np.nan, dtype=np.float64)
    ok = reliability > 0
    out[ok] = (r[ok] ** 2) / reliability[ok]
    return out


def summarise(r: np.ndarray, reliability: Optional[np.ndarray] = None) -> Dict[str, float]:
    s = {"median_r": float(np.median(r)), "mean_r": float(np.mean(r)), "n_neurons": int(len(r))}
    if reliability is not None:
        nr2 = normalised_r2(r, reliability)
        s["median_normalised_r2"] = float(np.nanmedian(nr2))
        s["median_ceiling"] = float(np.median(reliability))
    return s


def compute_rdm(representations: np.ndarray) -> np.ndarray:
    """Image-by-image dissimilarity, 1 minus Pearson correlation across features."""
    x = np.nan_to_num(representations.astype(np.float64))
    corr = np.corrcoef(x)
    corr = np.nan_to_num((corr + corr.T) / 2)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray):
    """Spearman correlation of the upper triangles. Returns (rho, p)."""
    mask = np.triu(np.ones_like(rdm_a, dtype=bool), k=1)
    rho, p = spearmanr(rdm_a[mask], rdm_b[mask])
    return float(rho), float(p)


def rdm_ceiling(responses_trials: np.ndarray, n_splits: int = 20, seed: int = 0) -> float:
    """Split-half reliability of the neural RDM itself, Spearman-Brown corrected.

    The right ceiling for an RDM comparison: how well the RDM from half the
    trials matches the RDM from the other half.
    """
    n_images, n_trials, _ = responses_trials.shape
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        a = np.nanmean(responses_trials[:, perm[: n_trials // 2], :], axis=1)
        b = np.nanmean(responses_trials[:, perm[n_trials // 2:], :], axis=1)
        rho, _ = compare_rdms(compute_rdm(a), compute_rdm(b))
        vals.append(2 * rho / (1 + rho))
    return float(np.mean(vals))
