"""
Improved decision engine:
- Prefer differential privacy / noise for numerical attributes.
- Prefer generalisation for Age-like attributes.
- Prefer k-anonymity / l-diversity / t-closeness for categorical quasi-identifiers and sensitive cols.
- Avoid suppression unless column(s) are near-unique identifiers.
- Produce deterministic, explainable decisions (technique + reason + params).
"""

def decide_techniques(profile_metrics: dict, pes_score=None, max_per_column=2):
    """
    profile_metrics: expected to be either:
      - a dict with per-column metrics under keys "columns" or "per_column",
      - or an aggregate dict with keys like 'uniqueness_ratio','entropy','dimensionality','outlier_index'.
    pes_score: numeric privacy exposure (higher -> more aggressive privacy).
    Returns: list[dict] with {'technique','reason','params'} ordered for application.
    """
    decisions = []
    seen = set()

    # Helper to add decision once
    def add(tech, reason, params=None):
        if tech in seen:
            return
        seen.add(tech)
        decisions.append({"technique": tech, "reason": reason, "params": params or {}})

    per_col = profile_metrics.get("columns") or profile_metrics.get("per_column") or {}
    # Column-level heuristics if available
    if isinstance(per_col, dict) and per_col:
        for col, m in per_col.items():
            col_low = col.lower()
            is_numeric = bool(m.get("is_numeric") or m.get("dtype") and str(m.get("dtype")).startswith("float") or m.get("dtype") and str(m.get("dtype")).startswith("int"))
            uniq = float(m.get("uniqueness_ratio") or m.get("uniqueness") or 0.0)
            ent = float(m.get("entropy") or 0.0)
            outl = float(m.get("outlier_index") or 0.0)

            # Age / year-like -> generalisation first
            if "age" in col_low or "year" in col_low:
                add("generalisation", "age_generalisation", {"bins": "auto"})
                # keep also a mild noise/DP to avoid exact values
                add("noise_addition", "post_generalisation_noise", {"scale": max(0.01, ent * 0.05)})
                continue

            if is_numeric:
                # numeric: DP preferred, add noise as fallback/augmentation
                # pick epsilon inversely proportional to pes_score / entropy: higher PES -> stronger privacy (smaller epsilon)
                base_eps = 0.5
                try:
                    if pes_score is not None:
                        base_eps = max(0.05, min(1.0, 1.0 - float(pes_score) * 0.5))
                    else:
                        base_eps = max(0.05, min(1.0, 1.0 - ent * 0.5))
                except Exception:
                    base_eps = 0.5
                add("differential_privacy", "numeric_dp_based_on_entropy_or_pes", {"epsilon": round(base_eps, 3)})
                add("noise_addition", "numeric_noise_for_robustness", {"scale": round(max(0.01, ent * 0.05), 4)})
                # handle outliers
                if outl > 0.4:
                    add("top_bottom_coding", "trim_extremes", {"lower_pct": 1, "upper_pct": 99})
                continue

            # categorical / string-like
            # very high uniqueness -> k-anonymity
            if uniq >= 0.5:
                k_val = min(10, max(2, int(uniq * 10)))
                add("k_anonymity", "high_uniqueness_categorical", {"k": k_val})
                # after k-anonymity, encourage l-diversity when sensitive distribution matters
                add("l_diversity", "improve_sensitive_diversity", {"l": 2})
            else:
                # moderate uniqueness: try l_diversity / t_closeness to preserve utility
                add("l_diversity", "protect_sensitive_values_with_low_uniqueness", {"l": 2})
                add("t_closeness", "preserve_distributional_similarity", {"t": 0.2})

            # fallback bucketization for mixed-type categorical that show numeric-like behavior
            if ent > 0.8:
                add("bucketization", "high_entropy_bucketize", {"n_buckets": 10})
    else:
        # Aggregate heuristics when no per-column info available
        uniq = float(profile_metrics.get("uniqueness_ratio", 0.0) or 0.0)
        ent = float(profile_metrics.get("entropy", 0.0) or 0.0)
        dim = float(profile_metrics.get("dimensionality", 0.0) or 0.0)
        outl = float(profile_metrics.get("outlier_index", 0.0) or 0.0)

        # If global uniqueness is high -> try k-anonymity (moderate k)
        if uniq >= 0.5:
            add("k_anonymity", "global_high_uniqueness", {"k": 5})
            add("l_diversity", "augment_k_with_l_diversity", {"l": 2})

        # numeric-heavy / high entropy -> DP + noise
        if ent >= 0.45 or dim >= 0.6:
            # choose epsilon based on PES if available
            eps = 0.5
            try:
                if pes_score is not None:
                    eps = max(0.05, min(1.0, 1.0 - float(pes_score) * 0.4))
                else:
                    eps = max(0.05, 1.0 - ent * 0.5)
            except Exception:
                eps = 0.5
            add("differential_privacy", "aggregate_high_entropy_or_dimensionality", {"epsilon": round(eps, 3)})
            add("noise_addition", "aggregate_noise", {"scale": round(max(0.01, ent * 0.05), 4)})

        # outliers -> bucket/top-bottom coding
        if outl > 0.4:
            add("top_bottom_coding", "global_trim_outliers", {"lower_pct": 1, "upper_pct": 99})

        # ensure at least one conservative technique
        if not decisions:
            add("noise_addition", "default_mild_noise", {"scale": 0.02})

    # Final ordering preference: preserve utility where possible
    preferred_order = [
        "generalisation", "k_anonymity", "l_diversity", "t_closeness",
        "bucketization", "top_bottom_coding", "noise_addition", "differential_privacy",
        "permutation", "suppression"
    ]
    decisions_sorted = sorted(decisions, key=lambda x: preferred_order.index(x["technique"]) if x["technique"] in preferred_order else len(preferred_order))
    return decisions_sorted
