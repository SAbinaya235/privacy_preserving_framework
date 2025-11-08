# profile.py — Dataset Nature Analysis & Metric Computation

"""
Implements quantitative profiling functions derived from:

El Emam, K. (2009). "A Systematic Review of Re-identification Attacks on Health Data."
Domingo-Ferrer, J. (2001). "A Quantitative Model for Data Privacy."
Used for profiling dataset nature and estimating privacy risk indicators.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy as scipy_entropy
from itertools import combinations


# ----------------------------
# 1. UNIQUENESS RATIO
# ----------------------------
def uniqueness_ratio(dataset):
    """
    Compute proportion of unique quasi-identifier combinations.
    Reference: El Emam (2009) – Re-identification risk metric.
    """
    quasi_identifiers = dataset.select_dtypes(include=['object', 'category', 'int', 'float'])
    if quasi_identifiers.empty:
        return 0.0
    duplicates = quasi_identifiers.duplicated()
    unique_count = (~duplicates).sum()
    ratio = unique_count / len(quasi_identifiers)
    return round(ratio, 3)


# ----------------------------
# 2. ENTROPY
# ----------------------------
def entropy(dataset):
    """
    Compute Shannon entropy H(X) for quasi-identifiers. low entropy - high risk of re-identification due to predictability.
    Reference: Shannon (1948); used for information content measurement.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number])
    total_entropy = 0
    for col in numeric_cols.columns:
        counts = dataset[col].value_counts(normalize=True)
        total_entropy += scipy_entropy(counts, base=2)
    avg_entropy = total_entropy / max(1, len(numeric_cols.columns))
    return round(avg_entropy, 3)


# ----------------------------
# 3. MUTUAL INFORMATION
# ----------------------------
def mutual_info(dataset):
    """
    Compute average mutual information between quasi-identifiers and sensitive attributes.
    Reference: Cover & Thomas (1991) – Information Theory.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number])
    categorical_cols = dataset.select_dtypes(include=['object', 'category'])

    if numeric_cols.empty or categorical_cols.empty:
        return 0.0

    scores = []
    for num_col in numeric_cols.columns:
        for cat_col in categorical_cols.columns:
            try:
                scores.append(mutual_info_score(dataset[num_col], dataset[cat_col]))
            except Exception:
                continue
    if not scores:
        return 0.0
    return round(np.mean(scores), 3)


# ----------------------------
# 4. KL DIVERGENCE
# ----------------------------
def kl_divergence(dataset):
    """
    Compute mean KL divergence between sensitive and quasi-identifier distributions.
    Reference: Li et al. (2007) – T-Closeness principle.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) < 2:
        return 0.0

    divergences = []
    cols = numeric_cols.columns
    for (c1, c2) in combinations(cols, 2):
        p = dataset[c1].value_counts(normalize=True)
        q = dataset[c2].value_counts(normalize=True)
        # align distributions
        p, q = p.align(q, fill_value=1e-9)
        divergences.append(scipy_entropy(p, q))
    if not divergences:
        return 0.0
    return round(np.mean(divergences), 3)


# ----------------------------
# 5. OUTLIER INDEX
# ----------------------------
def outlier_index(dataset):
    """
    Quantify outlier presence and intensity (using IQR rule).
    Reference: Tukey (1977) – Exploratory Data Analysis.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        return 0.0

    total_outliers = 0
    total_points = 0
    for col in numeric_cols.columns:
        q1 = dataset[col].quantile(0.25)
        q3 = dataset[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((dataset[col] < lower) | (dataset[col] > upper)).sum()
        total_outliers += outliers
        total_points += len(dataset[col])
    if total_points == 0:
        return 0.0
    return round(total_outliers / total_points, 3)


# ----------------------------
# 6. DIMENSIONALITY
# ----------------------------
def dimensionality(dataset):
    """
    Compute effective dimensionality (number of independent attributes).
    Uses correlation threshold to estimate redundancy.
    Reference: Jolliffe (2002) – Principal Component Analysis.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        return len(dataset.columns)

    corr = numeric_cols.corr().abs()
    redundant = (corr > 0.9).sum().sum() - len(corr)
    eff_dim = len(numeric_cols.columns) - redundant / max(1, len(numeric_cols.columns))
    return int(round(max(eff_dim, 1)))


# ----------------------------
# 7. PROFILE GENERATOR
# ----------------------------
def generate_profile(dataset):
    """Return a summary dictionary of all computed metrics."""
    profile = {
        "uniqueness_ratio": uniqueness_ratio(dataset),
        "entropy": entropy(dataset),
        "mutual_info": mutual_info(dataset),
        "kl_divergence": kl_divergence(dataset),
        "outlier_index": outlier_index(dataset),
        "dimensionality": dimensionality(dataset),
    }
    return profile
