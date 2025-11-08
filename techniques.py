# techniques.py
# -------------------------------------------------------------------
# Privacy-Preserving Transformation Library
# -------------------------------------------------------------------
# Contains the implementations (or simulation placeholders) of all
# privacy-preserving techniques supported by the framework.
#
# Each technique is implemented as a function that accepts a dataset,
# applies transformation logic, and returns a transformed dataset.
#
# References:
# - Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy.
# - Machanavajjhala et al. (2007). l-Diversity: Privacy Beyond k-Anonymity.
# - Li et al. (2007). t-Closeness: Privacy Beyond l-Diversity.
# - Dwork, C. (2006). Differential Privacy.
# - Domingo-Ferrer, J. (2001). Quantitative Model for Data Privacy.
# -------------------------------------------------------------------

import numpy as np
import pandas as pd

# ------------------------- Basic Perturbative Methods -------------------------

def suppression(dataset: pd.DataFrame, threshold: float = 0.05):
    """
    Suppress rare values or unique identifiers.
    Parameters:
        threshold: minimum frequency allowed for a value before suppression.
    """
    df = dataset.copy()
    for col in df.columns:
        freqs = df[col].value_counts(normalize=True)
        rare_values = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: np.nan if x in rare_values else x)
    return df


def generalisation(dataset: pd.DataFrame, hierarchies=None):
    """
    Replace specific values with broader categories (e.g., 23 -> 20-30).
    """
    df = dataset.copy()
    for col in df.select_dtypes(include=[np.number]):
        df[col] = pd.cut(df[col], bins=5, labels=False)  # coarse binning
    return df


def replacement(dataset: pd.DataFrame, placeholder="***"):
    """
    Replace sensitive attribute values with placeholders.
    """
    df = dataset.copy()
    for col in df.columns:
        if "name" in col.lower() or "id" in col.lower():
            df[col] = placeholder
    return df


def top_bottom_coding(dataset: pd.DataFrame, lower=5, upper=95):
    """
    Cap extreme values by percentile boundaries.
    """
    df = dataset.copy()
    for col in df.select_dtypes(include=[np.number]):
        low, high = np.percentile(df[col], [lower, upper])
        df[col] = np.clip(df[col], low, high)
    return df


def noise_addition(dataset: pd.DataFrame, epsilon=0.05):
    """
    Add random noise to numerical attributes.
    Smaller epsilon means stronger privacy but less utility.
    """
    df = dataset.copy()
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col] + np.random.normal(0, epsilon * df[col].std(), size=len(df))
    return df


def permutation(dataset: pd.DataFrame):
    """
    Randomly shuffle values within each column to break direct linkages.
    """
    df = dataset.copy()
    for col in df.columns:
        df[col] = np.random.permutation(df[col].values)
    return df


# ------------------------- Statistical Methods -------------------------

def k_anonymity(dataset: pd.DataFrame, k=None):
    """
    Enforce k-anonymity.
    Automatically determines k based on dataset size:
        - small datasets → smaller k (to avoid over-suppression)
        - large datasets → larger k
    """
    df = dataset.copy()
    n = len(df)
    if k is None:
        if n < 100:
            k = 3
        elif n < 1000:
            k = 5
        else:
            k = 10

    # Simplified enforcement: remove records that appear < k times
    qi_cols = df.columns[:min(3, len(df.columns))]  # assume first 3 as QIs
    grouped = df.groupby(list(qi_cols))
    mask = grouped.transform('size') >= k
    df = df[mask.all(axis=1)]
    df.reset_index(drop=True, inplace=True)
    return df


def l_diversity(dataset: pd.DataFrame, sensitive_col=None, l=2):
    """
    Ensure that each group has at least l distinct sensitive values.
    """
    df = dataset.copy()
    if sensitive_col is None:
        sensitive_col = df.columns[-1]
    qi_cols = df.columns[:-1]

    groups = df.groupby(list(qi_cols))
    valid_groups = []
    for _, g in groups:
        if g[sensitive_col].nunique() >= l:
            valid_groups.append(g)
    if valid_groups:
        df = pd.concat(valid_groups).reset_index(drop=True)
    return df


def t_closeness(dataset: pd.DataFrame, sensitive_col=None, t=0.2):
    """
    Reduce the distance between sensitive attribute distributions across groups.
    """
    df = dataset.copy()
    if sensitive_col is None:
        sensitive_col = df.columns[-1]
    overall_dist = df[sensitive_col].value_counts(normalize=True)
    qi_cols = df.columns[:-1]

    def group_distance(g):
        dist = g[sensitive_col].value_counts(normalize=True)
        kl = np.sum(dist * np.log((dist + 1e-9) / (overall_dist + 1e-9)))
        return kl

    groups = df.groupby(list(qi_cols))
    for name, g in groups:
        if group_distance(g) > t:
            df.loc[g.index, sensitive_col] = np.random.choice(df[sensitive_col])
    return df


def differential_privacy(dataset: pd.DataFrame, epsilon=None):
    """
    Apply differential privacy mechanism (Laplace noise addition).
    """
    df = dataset.copy()
    if epsilon is None:
        epsilon = 1.0 if len(df) > 1000 else 0.5

    sensitivity = 1.0
    scale = sensitivity / epsilon

    for col in df.select_dtypes(include=[np.number]):
        noise = np.random.laplace(0, scale, len(df))
        df[col] = df[col] + noise
    return df


# ------------------------- Utility Preserving / Hybrid -------------------------

def pseudonymization(dataset: pd.DataFrame):
    """
    Replace direct identifiers with pseudonyms.
    """
    df = dataset.copy()
    for col in df.columns:
        if "name" in col.lower() or "id" in col.lower():
            df[col] = [f"user_{i}" for i in range(len(df))]
    return df


def microaggregation(dataset: pd.DataFrame, group_size=5):
    """
    Replace individual records with group averages (useful for numerical data).
    """
    df = dataset.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].sort_values().reset_index(drop=True)
        df[col] = df[col].groupby(df.index // group_size).transform('mean')
    return df


def attribute_masking(dataset: pd.DataFrame, mask_prob=0.1):
    """
    Randomly mask a portion of attribute values.
    """
    df = dataset.copy()
    mask = np.random.rand(*df.shape) < mask_prob
    df = df.mask(mask)
    return df


def local_suppression(dataset: pd.DataFrame, outlier_threshold=0.95):
    """
    Suppress records with extreme attribute values (local suppression).
    """
    df = dataset.copy()
    for col in df.select_dtypes(include=[np.number]):
        high = np.percentile(df[col], outlier_threshold * 100)
        df.loc[df[col] > high, col] = np.nan
    return df


def pca_reduction(dataset: pd.DataFrame, n_components=0.9):
    """
    Apply PCA-based dimensionality reduction.
    n_components: fraction of variance to retain (default 90%).
    """
    from sklearn.decomposition import PCA
    num_cols = dataset.select_dtypes(include=[np.number]).columns
    df = dataset.copy()
    if len(num_cols) > 1:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(df[num_cols])
        reduced_df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(reduced.shape[1])])
        df = pd.concat([reduced_df, df.drop(columns=num_cols)], axis=1)
    return df


def minimal_masking(dataset: pd.DataFrame):
    """
    Apply light-weight tokenization for already safe datasets.
    """
    df = dataset.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: str(hash(x))[:6])
    return df


def t_closeness_adjustment(dataset: pd.DataFrame):
    """
    Wrapper for t-closeness with default parameters.
    """
    return t_closeness(dataset, t=0.3)
