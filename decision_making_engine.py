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

    # --- Ensure DP / Noise are applied if any numeric attributes exist ---
    numeric_present = False
    numeric_entropies = []
    if isinstance(per_col, dict) and per_col:
        for m in per_col.values():
            is_num = bool(m.get("is_numeric") or (m.get("dtype") and str(m.get("dtype")).startswith(("float","int"))))
            if is_num:
                numeric_present = True
                try:
                    numeric_entropies.append(float(m.get("entropy", 0.0)))
                except Exception:
                    pass
    else:
        # Fallback: treat as numeric-heavy if dimensionality/entropy high
        try:
            if float(profile_metrics.get("dimensionality", 0.0) or 0.0) >= 0.6 or float(profile_metrics.get("entropy", 0.0) or 0.0) >= 0.45:
                numeric_present = True
        except Exception:
            numeric_present = False

    if numeric_present:
        avg_ent = float(sum(numeric_entropies) / len(numeric_entropies)) if numeric_entropies else float(profile_metrics.get("entropy", 0.0) or 0.0)
        # choose epsilon inverse to PES / entropy (smaller epsilon = stronger privacy)
        eps = 0.5
        try:
            if pes_score is not None:
                eps = max(0.01, min(1.0, 1.0 - float(pes_score) * 0.4))
            else:
                eps = max(0.01, min(1.0, 1.0 - avg_ent * 0.5))
        except Exception:
            eps = 0.5
        scale = round(max(0.001, avg_ent * 0.05), 6)
        # force-add DP and noise (will be de-duplicated by add)
        add("differential_privacy", "numeric_dp_for_numericals_detected", {"epsilon": round(eps, 3)})
        add("noise_addition", "numeric_noise_for_numericals_detected", {"scale": scale})
    # --------------------------------------------------------------------

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
                # numeric: DP already ensured above; add numeric-specific robustness techniques
                add("noise_addition", "numeric_noise_for_robustness", {"scale": round(max(0.01, ent * 0.05), 4)})
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
            k_val = 5
            add("k_anonymity", "global_high_uniqueness", {"k": k_val})
            add("l_diversity", "augment_k_with_l_diversity", {"l": 2})

        # numeric-heavy / high entropy -> DP + noise already ensured above; add noise fallback if not present
        if (ent >= 0.45 or dim >= 0.6) and not any(d["technique"] == "differential_privacy" for d in decisions):
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

    # Ensure suppression is treated as a true last-resort:
    # - keep any detected 'suppression' decision aside,
    # - remove it from the main decisions list so other techniques are preferred,
    # - only re-add it if no other technique was chosen (fallback).
    suppression_decision = None
    for d in decisions[:]:
        if d.get("technique") == "suppression":
            suppression_decision = d
            decisions.remove(d)

    # If there are no other techniques selected, use suppression as a fallback.
    if not decisions and suppression_decision is not None:
        decisions.append(suppression_decision)

    decisions_sorted = sorted(decisions, key=lambda x: preferred_order.index(x["technique"]) if x["technique"] in preferred_order else len(preferred_order))
    return decisions_sorted
