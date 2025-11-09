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

import pandas as pd
import numpy as np

# ------------------------- Basic Perturbative Methods -------------------------

def suppression(dataset: pd.DataFrame, threshold: float = 0.05, columns: list | None = None, allow_numeric: bool = False):
    """
    Suppress (mask) rare values by replacing them with NaN.
    - columns: list of column names to restrict suppression (None => all columns).
    - allow_numeric: when True, numeric columns may be suppressed; default False.
    This enables per-attribute suppression experiments while preserving the
    default behavior of avoiding numeric-suppression unless explicitly requested.
    """
    df = dataset.copy()
    cols = list(columns) if columns else list(df.columns)
    for col in cols:
        if col not in df.columns:
            continue
        # skip numeric columns unless caller explicitly allows it
        if pd.api.types.is_numeric_dtype(df[col]) and not allow_numeric:
            continue

        try:
            vc = df[col].value_counts(normalize=True, dropna=True)
            rare = vc[vc < threshold].index
            df[col] = df[col].where(~df[col].isin(rare), other=np.nan)
        except Exception:
            # best-effort: if value_counts fails, skip column
            continue
    return df


def generalisation(dataset: pd.DataFrame, hierarchies=None, bins='auto'):
    """
    Generalise numeric attributes. Special-case 'Age' to human-friendly buckets.
    bins: 'auto', int, or list of bin edges.
    """
    df = dataset.copy()
    # Age special handling
    if "Age" in df.columns:
        try:
            edges = [0, 12, 18, 30, 45, 60, 75, 120]
            labels = ["0-11","12-17","18-29","30-44","45-59","60-74","75+"]
            df["Age"] = pd.cut(df["Age"], bins=edges, labels=labels, include_lowest=True)
        except Exception:
            # fallback: leave Age unchanged
            pass

    # general numeric binning for other numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c == "Age":
            continue
        try:
            if isinstance(bins, int):
                df[c] = pd.cut(df[c], bins=bins, duplicates="drop")
            elif isinstance(bins, (list, tuple)):
                df[c] = pd.cut(df[c], bins=bins, duplicates="drop")
            else:
                # automatic: use 10 bins or quantile-based if many unique
                uniq = df[c].nunique(dropna=True)
                nb = min(10, max(2, int(uniq**0.5)))
                df[c] = pd.qcut(df[c].rank(method="first"), q=nb, duplicates="drop")
        except Exception:
            # if binning fails, ignore and keep original
            continue
    return df


def replacement(dataset: pd.DataFrame, placeholder="***"):
    df = dataset.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(placeholder)
    return df


def top_bottom_coding(dataset: pd.DataFrame, lower=5, upper=95):
    df = dataset.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        try:
            lp = np.nanpercentile(df[c], lower)
            up = np.nanpercentile(df[c], upper)
            df[c] = df[c].clip(lower=lp, upper=up)
        except Exception:
            continue
    return df


def noise_addition(dataset: pd.DataFrame, scale=0.01, random_state=None):
    """
    Add small Gaussian noise to numeric columns. scale is relative to column std.
    scale: fraction of std to use as noise std (e.g., 0.01 means noise_std = 0.01 * col_std).
    """
    rng = np.random.default_rng(random_state)
    df = dataset.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        col = df[c]
        # skip columns with all NaN
        if col.dropna().empty:
            continue
        col_std = float(col.std(ddof=0)) if col.std(ddof=0) > 0 else float(np.abs(col.dropna()).mean() or 1.0)
        noise_std = max(1e-6, scale * col_std)
        noise = rng.normal(loc=0.0, scale=noise_std, size=len(col))
        # preserve NaNs
        df[c] = col + pd.Series(noise, index=col.index)
    return df


def permutation(dataset: pd.DataFrame, columns: list | None = None, random_state=None):
    """
    Permute values within specified columns independently to break linkage.
    - columns: list of column names to permute (None -> permute all columns).
    - random_state: optional seed for reproducibility.
    """
    df = dataset.copy()
    rng = np.random.default_rng(random_state)
    cols = list(columns) if columns is not None else df.columns.tolist()
    for c in cols:
        if c not in df.columns:
            continue
        try:
            seed = int(rng.integers(1_000_000))
            # sample and reassign values (preserves index length)
            df[c] = df[c].sample(frac=1.0, random_state=seed).values
        except Exception:
            # best-effort: skip column on failure
            continue
    return df


# ------------------------- Statistical Methods -------------------------

def k_anonymity(df, quasi_identifiers=None, k=5):
    """
    Enforce k-anonymity by keeping only rows that belong to groups of size >= k.
    quasi_identifiers: list of column names to consider (defaults to all columns).
    """
    if quasi_identifiers is None:
        quasi_identifiers = df.columns.tolist()

    try:
        group_sizes = df.groupby(quasi_identifiers).transform('size')
    except Exception:
        sizes_by_group = df.groupby(quasi_identifiers).size()
        idx = df.set_index(quasi_identifiers).index
        mapped = sizes_by_group.reindex(idx).to_numpy()
        mask = mapped >= k
        return df.iloc[mask].reset_index(drop=True)

    if hasattr(group_sizes, "ndim") and group_sizes.ndim == 2:
        sizes_series = group_sizes.iloc[:, 0]
    else:
        sizes_series = group_sizes

    mask = (sizes_series >= k)
    return df.iloc[mask.to_numpy()].reset_index(drop=True)


def l_diversity(dataset: pd.DataFrame, sensitive_col=None, l=2):
    df = dataset.copy()
    if sensitive_col is None:
        # try to pick a likely sensitive column
        for cand in ["Diagnosis", "Treatment_Type", "Bill_Amount"]:
            if cand in df.columns:
                sensitive_col = cand
                break
    if sensitive_col is None or sensitive_col not in df.columns:
        return df

    # Keep groups where the sensitive_col has at least l distinct values
    qids = [c for c in df.columns if c != sensitive_col]
    if not qids:
        return df
    sizes = df.groupby(qids)[sensitive_col].nunique()
    valid = sizes[sizes >= l].index
    try:
        mask = df.set_index(qids).index.isin(valid)
        return df[mask].reset_index(drop=True)
    except Exception:
        return df


def t_closeness(dataset: pd.DataFrame, sensitive_col=None, t=0.2):
    # Simple placeholder: if distribution of sensitive_col within groups differs a lot, apply no-op.
    return dataset.copy()


def differential_privacy(dataset: pd.DataFrame, epsilon=0.5, random_state=None):
    """
    Apply Laplace mechanism to numeric columns. Noise magnitude is bounded to avoid obliterating data.
    Epsilon controls noise (smaller -> stronger privacy). We cap noise scale conservatively.
    """
    rng = np.random.default_rng(random_state)
    df = dataset.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    n = max(1, len(df))
    for c in num_cols:
        col = df[c]
        if col.dropna().empty:
            continue
        # estimate sensitivity: use range or std as proxy, scaled down to avoid extreme noise
        col_min = float(np.nanmin(col))
        col_max = float(np.nanmax(col))
        rng_col = max(1e-6, col_max - col_min)
        col_std = float(col.std(ddof=0)) if col.std(ddof=0) > 0 else (abs(col.dropna()).mean() or 1.0)
        # sensitivity proxy (moderate): use smaller of range and (2*std)
        sensitivity = max(1e-6, min(rng_col, 2.0 * col_std))
        # scale for Laplace: sensitivity / epsilon, but cap by a fraction of data range to avoid destroying signal
        eps = max(1e-3, float(epsilon))
        raw_scale = sensitivity / eps
        cap = max(1.0, rng_col * 0.25)  # never allow noise std >> 25% of range
        scale = min(raw_scale, cap)
        noise = rng.standard_cauchy(size=len(col))  # heavy-tailed fallback if needed
        # Use Laplace-like by sampling from laplace
        noise = rng.laplace(loc=0.0, scale=scale, size=len(col))
        df[c] = col + pd.Series(noise, index=col.index)
    return df


# ------------------------- Utility Preserving / Hybrid -------------------------

def pseudonymization(dataset: pd.DataFrame):
    df = dataset.copy()
    # simple pseudonymize identifiable string columns by hashing
    for c in df.select_dtypes(include=[object]).columns:
        try:
            df[c] = df[c].fillna("").apply(lambda v: "id_" + str(abs(hash(v)) % 1000000) if v != "" else v)
        except Exception:
            continue
    return df


def microaggregation(dataset: pd.DataFrame, group_size=5):
    df = dataset.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    # cluster by rounding and replace groups by cluster mean (simple)
    for c in num_cols:
        try:
            df[c] = df[c].groupby((df[c].round(-1)).fillna(method='ffill')).transform("mean")
        except Exception:
            continue
    return df


def attribute_masking(dataset: pd.DataFrame, mask_prob=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    df = dataset.copy()
    for c in df.columns:
        if df[c].dtype == object:
            mask = rng.random(len(df)) < mask_prob
            df.loc[mask, c] = np.nan
    return df


def local_suppression(dataset: pd.DataFrame, outlier_threshold=0.95):
    df = dataset.copy()
    # suppress extreme values in numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        q = df[c].quantile(outlier_threshold)
        df.loc[df[c] > q, c] = np.nan
    return df


def pca_reduction(dataset: pd.DataFrame, n_components=0.9):
    # placeholder: do nothing if sklearn not available
    return dataset.copy()


def minimal_masking(dataset: pd.DataFrame):
    df = dataset.copy()
    # replace names with initials
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).apply(lambda x: (x.split()[0][0] + ".") if x else x)
    return df


def t_closeness_adjustment(dataset: pd.DataFrame):
    return dataset.copy()


def bucketization(dataset: pd.DataFrame, n_buckets: int = 10, columns=None, strategy: str = 'equal_width', labels: bool = False):
    df = dataset.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for c in cols:
        try:
            if strategy == 'quantile':
                df[c] = pd.qcut(df[c].rank(method="first"), q=n_buckets, duplicates="drop")
            else:
                df[c] = pd.cut(df[c], bins=n_buckets, duplicates="drop")
            if not labels:
                df[c] = df[c].astype(str)
        except Exception:
            continue
    return df
