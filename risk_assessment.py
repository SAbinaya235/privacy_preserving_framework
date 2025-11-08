# Quantitative Privacy Exposure Computation
"""
Reference:
Domingo-Ferrer, J. (2001). "A Quantitative Model for Data Privacy."
Complemented with concepts from:
Samarati & Sweeney (1998) – K-Anonymity
Li et al. (2007) – T-Closeness
El Emam (2009) – Re-identification Risk Analysis

Purpose:
Converts dataset profiling metrics into a composite Privacy Exposure Score (PES)
that indicates how much privacy risk a dataset presents prior to any protection.
"""

import numpy as np

def calculate_pes(profile_metrics):
    """
    Compute Privacy Exposure Score (PES) from profiling metrics.

    Each profiling metric contributes to the final score with a specific weight.
    The result is normalized to [0, 1], where:
        0 → fully privacy-preserved
        1 → highly privacy-exposed

    Input:
        profile_metrics (dict): output from profile.generate_profile()

    Returns:
        dict: {
            "PES": float,
            "risk_level": str
        }
    """

    # ----- 1. Extract metrics -----
    u = profile_metrics.get("uniqueness_ratio", 0)
    e = profile_metrics.get("entropy", 0)
    m = profile_metrics.get("mutual_info", 0)
    k = profile_metrics.get("kl_divergence", 0)
    o = profile_metrics.get("outlier_index", 0)
    d = profile_metrics.get("dimensionality", 1)

    # ----- 2. Normalization -----
    # Entropy and dimensionality are inverted — higher values often reduce re-identification risk.
    # Uniqueness, mutual info, KL divergence, and outliers increase risk.
    # Normalization ensures metrics contribute comparably.
    norm = {
        "u": u,  # already between [0,1]
        "e": 1 / (1 + np.exp(-e)),  # sigmoid to bound entropy
        "m": np.tanh(m),
        "k": np.tanh(k),
        "o": min(o * 5, 1.0),
        "d": 1 / (1 + d)  # more dimensions → lower exposure
    }

    # ----- 3. Weighted Aggregation -----
    # Based on sensitivity of each metric to disclosure risk.
    weights = {
        "u": 0.25,  # uniqueness strongly increases re-identification
        "e": 0.15,  # entropy contributes moderately
        "m": 0.20,  # linkability
        "k": 0.15,  # distributional deviation
        "o": 0.15,  # outliers
        "d": 0.10   # dimensionality effect (inverse)
    }

    pes = (
        weights["u"] * norm["u"] +
        weights["e"] * (1 - norm["e"]) +  # inverse effect
        weights["m"] * norm["m"] +
        weights["k"] * norm["k"] +
        weights["o"] * norm["o"] +
        weights["d"] * norm["d"]
    )

    pes = round(pes, 3)

    # ----- 4. Risk Level Categorization -----
    if pes < 0.33:
        level = "Low"
    elif pes < 0.66:
        level = "Medium"
    else:
        level = "High"

    return {"PES": pes, "risk_level": level}
