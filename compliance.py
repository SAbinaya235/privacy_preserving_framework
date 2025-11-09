# GDPR Compliance Verification with Remediation Actions
# References: GDPR Articles 5(1)(b–e), 6, 17, 25

import pandas as pd
import numpy as np
import uuid
from typing import Tuple, List, Dict, Any

GDPR_ARTICLES = {
    "purpose_limitation": {"article": "Article 5(1)(b)", "description": "Data used only for declared purpose"},
    "data_minimization": {"article": "Article 5(1)(c)", "description": "Limit data to what is necessary"},
    "lawfulness": {"article": "Article 6", "description": "Lawful basis for processing"},
    "erasure_right": {"article": "Article 17", "description": "Right to erasure of data"},
    "privacy_by_design": {"article": "Article 25", "description": "Integrate privacy controls in design"}
}

# ------------------------------------------------------------------------------

def check_purpose_limitation(metadata):
    """Ensure declared data purpose matches actual use."""
    declared = metadata.get("purpose")
    actual = metadata.get("use_case")
    return declared == actual

def check_data_minimization(dataset):
    """Ensure dataset contains only required attributes."""
    required_columns = ["age", "gender", "income"]  # Example required set
    return all(col in required_columns for col in dataset.columns)

def check_lawfulness(dataset):
    """Verify that data collection has a lawful basis."""
    # In a real case: check metadata for consent/contract/etc.
    return "consent_flag" in dataset.columns

def check_erasure_right(dataset):
    """Check if records have unique IDs to support deletion."""
    return "record_id" in dataset.columns

def check_privacy_by_design(metadata):
    """Check presence of design-level privacy features."""
    return metadata.get("encryption_enabled", False)

# ------------------------------------------------------------------------------

def perform_compliance_check(dataset, metadata):
    """Run all checks and return compliance report with remediation actions."""

    check_functions = {
        "purpose_limitation": check_purpose_limitation,
        "data_minimization": check_data_minimization,
        "lawfulness": check_lawfulness,
        "erasure_right": check_erasure_right,
        "privacy_by_design": check_privacy_by_design
    }

    # Run all compliance checks
    report = []
    passed_count = 0

    for key, func in check_functions.items():
        result = func(dataset if "dataset" in func.__code__.co_varnames else metadata)
        passed = bool(result)
        if passed:
            passed_count += 1
        report.append({
            "check": key,
            "status": "Pass" if passed else "Fail",
            "article": GDPR_ARTICLES[key]["article"],
            "description": GDPR_ARTICLES[key]["description"]
        })

    # --------------------------------------------------------------------------
    # ✅ Add Remediation Actions
    REMEDIATION_GUIDE = {
        "purpose_limitation": "Align dataset usage with declared purpose in metadata.",
        "data_minimization": "Remove or anonymize unnecessary personal data fields.",
        "lawfulness": "Specify a lawful basis for data processing (e.g., consent or contract).",
        "erasure_right": "Add unique identifiers to enable data deletion requests.",
        "privacy_by_design": "Integrate technical controls such as encryption or pseudonymization."
    }

    remediation_actions = [
        
        {
            "article": r["article"],
            "action": REMEDIATION_GUIDE[r["check"]]
        }
        for r in report if r["status"] == "Fail"
    ]

    # --------------------------------------------------------------------------
    # Compute compliance score
    compliance_score = passed_count / len(check_functions)

    final_report = {
        "checks": report,
        "compliance_score": round(compliance_score, 2),
        "compliance_status": "Compliant" if compliance_score > 0.8 else "Partially Compliant",
        "remediation_actions": remediation_actions
    }

    return final_report

# Heuristic helpers (existing or re-used from earlier suggestions)
def _has_obvious_pii(df: pd.DataFrame) -> float:
    if df is None or df.shape[0] == 0:
        return 1.0
    n = len(df)
    pii_indicators = ["id", "name", "ssn", "email", "patient", "phone", "device"]
    suspicious_scores = []
    for col in df.columns:
        col_low = col.lower()
        if any(tok in col_low for tok in pii_indicators):
            try:
                uniq_frac = df[col].nunique(dropna=True) / max(1, n)
                suspicious_scores.append(min(1.0, uniq_frac))
            except Exception:
                suspicious_scores.append(0.5)
    if not suspicious_scores:
        return 1.0
    max_susp = max(suspicious_scores)
    return float(max(0.0, min(1.0, 1.0 - max_susp)))

def _estimate_k_compliance(profile_metrics: dict, desired_k: int = 10) -> float:
    uniq = float(profile_metrics.get("uniqueness_ratio", 0.0) or 0.0)
    avg_group_size = 1.0 / max(uniq, 1.0 / (desired_k * 10.0)) if uniq > 0 else desired_k * 2
    score = min(1.0, avg_group_size / float(desired_k))
    return float(np.clip(score, 0.0, 1.0))

def perform_compliance_check(transformed_df: pd.DataFrame, profile_for_check: dict | None = None, desired_k: int = 10) -> dict:
    profile = profile_for_check or {}
    k_comp = _estimate_k_compliance(profile, desired_k=desired_k)
    no_res_pii = _has_obvious_pii(transformed_df)
    # Basic rule checks (pass/fail)
    checks = []
    checks.append({"check": "purpose_limitation", "status": "Pass", "article": GDPR_ARTICLES["purpose_limitation"]["article"], "description": GDPR_ARTICLES["purpose_limitation"]["description"]})
    # data minimization heuristic
    checks.append({"check": "data_minimization", "status": "Pass" if no_res_pii >= 0.9 else "Fail", "article": GDPR_ARTICLES["data_minimization"]["article"], "description": GDPR_ARTICLES["data_minimization"]["description"]})
    checks.append({"check": "lawfulness", "status": "Pass" if profile.get("lawful_basis") else "Fail", "article": GDPR_ARTICLES["lawfulness"]["article"], "description": GDPR_ARTICLES["lawfulness"]["description"]})
    checks.append({"check": "erasure_right", "status": "Pass" if "deletion_id" in (transformed_df.columns if transformed_df is not None else []) else "Fail", "article": GDPR_ARTICLES["erasure_right"]["article"], "description": GDPR_ARTICLES["erasure_right"]["description"]})
    checks.append({"check": "privacy_by_design", "status": "Pass" if profile.get("privacy_by_design") else "Fail", "article": GDPR_ARTICLES["privacy_by_design"]["article"], "description": GDPR_ARTICLES["privacy_by_design"]["description"]})

    # Remediation suggestions
    remediation_actions = []
    if checks[1]["status"] == "Fail":
        remediation_actions.append({"article": GDPR_ARTICLES["data_minimization"]["article"], "action": "Remove or anonymize unnecessary personal data fields."})
    if checks[2]["status"] == "Fail":
        remediation_actions.append({"article": GDPR_ARTICLES["lawfulness"]["article"], "action": "Specify a lawful basis for data processing (e.g., consent or contract)."})
    if checks[3]["status"] == "Fail":
        remediation_actions.append({"article": GDPR_ARTICLES["erasure_right"]["article"], "action": "Add unique identifiers to enable data deletion requests."})
    if checks[4]["status"] == "Fail":
        remediation_actions.append({"article": GDPR_ARTICLES["privacy_by_design"]["article"], "action": "Integrate technical controls such as pseudonymization/encryption."})

    compliance_score = round(float((sum(1 for c in checks if c["status"] == "Pass") / len(checks))), 4)
    compliance_status = "Compliant" if compliance_score == 1.0 else ("Partially Compliant" if compliance_score > 0 else "Non-Compliant")

    return {
        "checks": checks,
        "compliance_score": compliance_score,
        "compliance_status": compliance_status,
        "remediation_actions": remediation_actions
    }

def perform_remediation(df: pd.DataFrame, compliance_report: dict, profile_metrics: dict | None = None) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:
    """
    Apply targeted remediation transformations based on the compliance_report.
    Returns (transformed_df, applied_actions)
    """
    from techniques import pseudonymization, suppression  # local import to avoid circular issues
    applied_actions = []
    new_df = df.copy()

    # 1) Data minimization: remove or anonymize commonly unnecessary PII columns
    if any(r["article"] == GDPR_ARTICLES["data_minimization"]["article"] for r in (compliance_report.get("remediation_actions") or [])):
        candidates_to_remove = ["Name", "Email", "Phone", "Address", "Patient_ID", "User_ID", "Device_ID"]
        removed = []
        for c in candidates_to_remove:
            if c in new_df.columns:
                # Prefer pseudonymization for ID-like cols rather than dropping if feasible
                if c.lower().endswith("id"):
                    try:
                        new_df[c] = new_df[c].astype(str).fillna("").apply(lambda v: "id_" + str(abs(hash(v)) % 1000000) if v != "" else "")
                        removed.append(f"pseudonymized:{c}")
                    except Exception:
                        new_df = new_df.drop(columns=[c])
                        removed.append(f"dropped:{c}")
                else:
                    new_df = new_df.drop(columns=[c])
                    removed.append(f"dropped:{c}")
        if removed:
            applied_actions.append({"article": GDPR_ARTICLES["data_minimization"]["article"], "actions": removed})

    # 2) Lawfulness: cannot create legal basis programmatically, but record a placeholder in profile/metadata
    if any(r["article"] == GDPR_ARTICLES["lawfulness"]["article"] for r in (compliance_report.get("remediation_actions") or [])):
        # signal that a lawful basis was chosen (human action required)
        applied_actions.append({"article": GDPR_ARTICLES["lawfulness"]["article"], "actions": ["metadata_set:lawful_basis='consent' (placeholder)"]})

    # 3) Erasure_right: add deletion_id column to support deletion requests
    if any(r["article"] == GDPR_ARTICLES["erasure_right"]["article"] for r in (compliance_report.get("remediation_actions") or [])):
        if "deletion_id" not in new_df.columns:
            new_df["deletion_id"] = [str(uuid.uuid4()) for _ in range(len(new_df))]
            applied_actions.append({"article": GDPR_ARTICLES["erasure_right"]["article"], "actions": ["added:deletion_id"]})

    # 4) Privacy by design: apply pseudonymization on identifier-like columns and set flag
    if any(r["article"] == GDPR_ARTICLES["privacy_by_design"]["article"] for r in (compliance_report.get("remediation_actions") or [])):
        try:
            new_df = pseudonymization(new_df)
            applied_actions.append({"article": GDPR_ARTICLES["privacy_by_design"]["article"], "actions": ["pseudonymized identificiable text columns"]})
        except Exception:
            pass

    # 5) Final small tidy: replace overly-unique free-text with suppression (best-effort)
    try:
        new_df = suppression(new_df, threshold=0.01, columns=None, allow_numeric=False)
    except Exception:
        pass

    return new_df, applied_actions
