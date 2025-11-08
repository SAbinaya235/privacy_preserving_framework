# Core Intelligence Unit – Technique Selection Logic

# Purpose:
# Implements the rules-based mapping engine that selects privacy-preserving techniques based on metrics and PES.


def decide_techniques(profile_metrics, pes_score):
    """
    Determine suitable privacy-preserving methods using rule-based logic.
    Returns a list of techniques with rationale.
    """
    # Example structure
    decisions = []
    
    # Sample rule (to be expanded later)
    if profile_metrics["uniqueness_ratio"] > 0.4:
        decisions.append({"technique": "generalisation", "reason": "High uniqueness ratio"})
    
    if pes_score > 0.5:
        decisions.append({"technique": "k_anonymity", "reason": "High privacy exposure"})

    return decisions


# bridge between profiling to transformation