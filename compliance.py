# GDPR Compliance Verification with Remediation Actions
# References: GDPR Articles 5(1)(b–e), 6, 17, 25

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
