"""
Mapping model features to neural responses: ceiling, readouts, scoring, RDMs.
Numpy, scipy and scikit-learn only, so it can be tested without torch.

Array conventions:
  responses_trials  (n_images, n_trials, n_neurons), NaN where a trial is missing
  responses         (n_images, n_neurons), the trial mean
  features          (n_images, n_features)
"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

RIDGE_ALPHAS = np.logspace(-2, 5, 8)


def split_half_reliability(responses_trials: np.ndarray, n_splits: int = 20, seed: int = 0) -> np.ndarray:
    """Per-neuron noise ceiling: split the trials in half, correlate the two half-means
    across images, Spearman-Brown correct (2r / (1 + r)), average over n_splits splits."""
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
    """Sparse random projection to n_components (Conwell et al. 2021), so whole feature maps
    can be used without PCA. Smaller feature sets are returned unchanged."""
    if features.shape[1] <= n_components:
        return features.astype(np.float32)
    proj = SparseRandomProjection(n_components=n_components, random_state=seed)
    return proj.fit_transform(features).astype(np.float32)


def cv_ridge_scores(features: np.ndarray, responses: np.ndarray, n_folds: int = 5, seed: int = 0,
                    alphas: np.ndarray = RIDGE_ALPHAS) -> Dict[str, np.ndarray]:
    """K-fold ridge from features to every neuron; one penalty per neuron, chosen by
    leave-one-out on the training fold. Held-out predictions are gathered across folds
    and scored once per neuron, which is steadier than averaging per-fold correlations.

    Returns {'predictions': (n_images, n_neurons), 'r': (n_neurons,)}.
    With uninformative features r comes out slightly negative (about -0.1 at 118
    images), because each fold regresses toward its own training mean.
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
    """r² of the prediction over the neuron's reliability. Undefined where the reliability
    is not positive; not clipped, so an underestimated ceiling stays visible as a value above 1."""
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
    """Ceiling for RDM comparisons: how well the RDM from half the trials matches the RDM
    from the other half, Spearman-Brown corrected."""
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


class RidgeReadout:
    """The usual readout: standardise, then ridge with one penalty per neuron."""
    name = "ridge"

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = alphas

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.model = RidgeCV(alphas=self.alphas, alpha_per_target=True).fit(self.scaler.transform(x), y)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


class LowRankReadout:
    """Ridge on the top k principal components: a readout with k free directions instead of
    one per feature."""
    def __init__(self, k=20, alphas=RIDGE_ALPHAS):
        self.k, self.alphas = k, alphas
        self.name = f"ridge_pca{k}"

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.pca = PCA(n_components=min(self.k, x.shape[0] - 1, x.shape[1]), random_state=0).fit(self.scaler.transform(x))
        z = self.pca.transform(self.scaler.transform(x))
        self.model = RidgeCV(alphas=self.alphas, alpha_per_target=True).fit(z, y)
        return self

    def predict(self, x):
        return self.model.predict(self.pca.transform(self.scaler.transform(x)))


def rsa_score(features: np.ndarray, responses: np.ndarray) -> float:
    """Spearman rho between the model RDM and the neural RDM. Nothing is fitted; one number
    for the image set rather than one per neuron."""
    rho, _ = compare_rdms(compute_rdm(features), compute_rdm(responses))
    return rho


def regression_readouts():
    """The fitted readouts, most to least flexible."""
    return [RidgeReadout(), LowRankReadout(k=100), LowRankReadout(k=20), LowRankReadout(k=5)]
