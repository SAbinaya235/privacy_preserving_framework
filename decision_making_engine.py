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
    Decide which privacy techniques to apply based on profiling metrics and PES score.
    Uses a tiered rule-based decision model for interpretability.
    """
    decisions = []

    # 🔹 Tier 1: Re-identification risk (uniqueness)
    if profile_metrics["uniqueness_ratio"] > 0.7:
        decisions.append({
            "technique": "generalisation",
            "reason": f"High uniqueness ratio ({profile_metrics['uniqueness_ratio']:.2f}) "
                      f"indicates re-identification risk; reduce granularity."
        })

    # 🔹 Tier 2: Moderate privacy exposure → mild noise
    if 0.4 <= pes_score <= 0.7:
        decisions.append({
            "technique": "noise_addition",
            "reason": f"PES = {pes_score:.2f} (Medium) → add slight statistical noise to safeguard sensitive data."
        })

    # 🔹 Tier 3: High privacy exposure → strong anonymization
    if pes_score > 0.7:
        decisions.extend([
            {
                "technique": "k_anonymity",
                "reason": f"PES = {pes_score:.2f} (High) → apply k-anonymity to ensure indistinguishable records."
            },
            {
                "technique": "pseudonymization",
                "reason": "Direct identifiers must be replaced to prevent identity linkage."
            }
        ])

    # 🔹 Tier 4: Attribute imbalance → distributional privacy
    if profile_metrics["kl_divergence"] > 10:
        decisions.append({
            "technique": "t_closeness",
            "reason": f"KL divergence ({profile_metrics['kl_divergence']:.2f}) "
                      f"shows sensitive attribute imbalance across equivalence classes."
        })

    return decisions
