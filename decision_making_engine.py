# Core Intelligence Unit – Technique Selection Logic
"""
Purpose:
Implements the rules-based mapping engine that selects privacy-preserving techniques
based on quantitative metrics and Privacy Exposure Score (PES).

References:
- Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy.
- Machanavajjhala et al. (2007). l-Diversity: Privacy Beyond k-Anonymity.
- Li et al. (2007). t-Closeness: Privacy Beyond l-Diversity.
- Domingo-Ferrer (2001). Quantitative Model for Data Privacy.
- El Emam (2009). Re-identification Risk Review.

The DME maps the profiling and PES results to appropriate privacy-preserving transformations
in a “clinical diagnosis” style:
    Dataset profile → Associated risk → Matching technique.
"""

def decide_techniques(profile_metrics, pes_score):
    """
    Determine suitable privacy-preserving methods using rule-based logic.

    Input:
        profile_metrics (dict): Quantitative dataset metrics from profiling.
        pes_score (float): Normalized privacy exposure score [0, 1].

    Output:
        list[dict]: [{"technique": str, "reason": str}]
    """

    decisions = []
    uniq = profile_metrics.get("uniqueness_ratio", 0)
    mi = profile_metrics.get("mutual_info", 0)
    kl = profile_metrics.get("kl_divergence", 0)
    out = profile_metrics.get("outlier_index", 0)
    dim = profile_metrics.get("dimensionality", 1)

    # --- Rule 1: High Uniqueness → Generalization / Suppression ---
    if uniq > 0.5:
        decisions.append({
            "technique": "generalisation",
            "reason": f"High uniqueness ratio ({uniq:.2f}) indicates re-identification risk; reduce granularity."
        })
    elif uniq > 0.3:
        decisions.append({
            "technique": "microaggregation",
            "reason": f"Moderate uniqueness ratio ({uniq:.2f}); cluster similar records to reduce identifiability."
        })

    # --- Rule 2: High Mutual Information → Perturbation ---
    if mi > 0.5:
        decisions.append({
            "technique": "noise_addition",
            "reason": f"High mutual information ({mi:.2f}) shows strong linkage between identifiers and sensitive attributes."
        })
    elif mi > 0.3:
        decisions.append({
            "technique": "attribute_masking",
            "reason": f"Moderate linkage detected; selective masking of attributes recommended."
        })

    # --- Rule 3: High KL Divergence → t-Closeness / Resampling ---
    if kl > 0.4:
        decisions.append({
            "technique": "t_closeness_adjustment",
            "reason": f"KL divergence ({kl:.2f}) indicates sensitive attribute distribution differs across groups."
        })

    # --- Rule 4: Outlier Risk → Suppression or Local Smoothing ---
    if out > 0.3:
        decisions.append({
            "technique": "local_suppression",
            "reason": f"Outlier index ({out:.2f}) high; rare records may reveal individuals."
        })

    # --- Rule 5: Dimensionality Effect → Dimensionality Reduction ---
    if dim > 10:
        decisions.append({
            "technique": "pca_reduction",
            "reason": f"High dimensionality ({dim}); apply PCA to remove redundant quasi-identifiers."
        })

    # --- Rule 6: PES-Based Global Decision ---
    if pes_score >= 0.66:
        decisions.append({
            "technique": "k_anonymity",
            "reason": f"PES = {pes_score:.2f} (High); enforce minimum k-anonymity threshold."
        })
        decisions.append({
            "technique": "l_diversity",
            "reason": "High exposure; ensure intra-group diversity for sensitive attributes."
        })
    elif 0.33 <= pes_score < 0.66:
        decisions.append({
            "technique": "pseudonymization",
            "reason": f"PES = {pes_score:.2f} (Medium); replace direct identifiers."
        })
    else:
        decisions.append({
            "technique": "minimal_masking",
            "reason": f"PES = {pes_score:.2f} (Low); apply lightweight tokenization only."
        })

    # --- Rule 7: Resolve Redundancies ---
    # Remove duplicate technique suggestions
    unique_decisions = []
    seen = set()
    for d in decisions:
        if d["technique"] not in seen:
            unique_decisions.append(d)
            seen.add(d["technique"])

    return unique_decisions
