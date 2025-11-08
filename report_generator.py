# Unified Logging and Report Synthesis
# -----------------------------------
# Collects intermediate results from each stage of the framework pipeline
# and compiles them into a structured final report.

import json
import os
import datetime

LOG_FILE = "logs/system_log.json"
FINAL_REPORT = "logs/final_report.json"

def log_step(step_name, data):
    """
    Enhanced logger – captures reasoning and evidence for applied techniques.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()

    entry = {
        "step": step_name,
        "timestamp": timestamp,
        "data": data
    }

    # Ensure log file exists
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump({"log_entries": []}, f, indent=4)

    # Append
    with open(LOG_FILE, "r+") as f:
        log_data = json.load(f)
        log_data["log_entries"].append(entry)
        f.seek(0)
        json.dump(log_data, f, indent=4)
        f.truncate()


def generate_final_report():
    """
    Generate a full interpretive final report.
    Includes evidence for why each technique was applied.
    """
    if not os.path.exists(LOG_FILE):
        print("No logs found to generate a report.")
        return

    with open(LOG_FILE, "r") as f:
        log_data = json.load(f)

    report = {
        "report_generated_at": datetime.datetime.now().isoformat(),
        "summary": {},
        "detailed_log": log_data["log_entries"]
    }

    # Extract
    for entry in log_data["log_entries"]:
        step = entry["step"]
        report["summary"][step] = entry["data"]

    # ✅ Enhanced Technique Evidence Mapping
    if "profiling" in report["summary"] and "decision_making" in report["summary"]:
        profiling = report["summary"]["profiling"]
        decisions = report["summary"]["decision_making"]

        technique_evidence = []
        for d in decisions:
            evidence = {
                "technique": d["technique"],
                "identified_problem": _map_problem(d["technique"], profiling),
                "reason": d["reason"],
                "expected_effect": _expected_effect(d["technique"]),
            }
            technique_evidence.append(evidence)
        report["technique_evidence"] = technique_evidence

    # Attach compliance and insights
    _attach_insights(report)

    # Save
    os.makedirs("logs", exist_ok=True)
    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Final Privacy Validation Report Generated Successfully!")
    print(f"📄 Location: {FINAL_REPORT}")
    print_report_summary(report)


def _map_problem(technique, metrics):
    """
    Maps technique to identified problem in dataset profile.
    """
    if technique == "generalisation":
        return f"High uniqueness ratio ({metrics.get('uniqueness_ratio')}) indicates possible identity disclosure."
    elif technique == "k_anonymity":
        return f"High privacy exposure suggests re-identification risk."
    elif technique == "l_diversity":
        return f"Sensitive attribute distribution imbalance — possible attribute disclosure."
    elif technique == "t_closeness":
        return f"KL divergence ({metrics.get('kl_divergence')}) indicates distance between groups’ sensitive values."
    elif technique == "suppression":
        return "Outlier index high — few records identifiable due to extreme values."
    elif technique == "noise_addition":
        return "High mutual information — attribute correlation might reveal private patterns."
    elif technique == "permutation":
        return "Correlated quasi-identifiers — linkage risk detected."
    elif technique == "differential_privacy":
        return "Overall high PES score, strong mathematical protection required."
    else:
        return "No specific problem mapping."


def _expected_effect(technique):
    """
    Describes the intended privacy impact of each technique.
    """
    mapping = {
        "generalisation": "Reduce identity disclosure by broadening attribute categories.",
        "suppression": "Remove identifiable records or rare quasi-identifier values.",
        "k_anonymity": "Ensure each record indistinguishable within k-group.",
        "l_diversity": "Protect against sensitive attribute inference.",
        "t_closeness": "Balance sensitive attribute distribution within groups.",
        "noise_addition": "Obfuscate exact values while retaining statistical trends.",
        "permutation": "Break linkage between identifiers and sensitive data.",
        "differential_privacy": "Guarantee privacy mathematically across all outputs."
    }
    return mapping.get(technique, "Enhances privacy protection.")


def _attach_insights(report):
    """
    Adds compliance and risk interpretation summary.
    """
    summary = report["summary"]
    pes = summary.get("risk_assessment", {}).get("PES", None)
    compliance_score = summary.get("compliance", {}).get("compliance_score", None)

    report["insights"] = {
        "initial_privacy_exposure": pes,
        "overall_compliance_score": compliance_score,
        "remarks": _generate_remark(pes, compliance_score)
    }


def _generate_remark(risk, compliance_score):
    """
    Generate textual remark based on privacy risk and compliance.
    """
    if risk is None or compliance_score is None:
        return "Insufficient data for full interpretation."
    
    if risk < 0.3 and compliance_score >= 0.8:
        return "Dataset is safe for use and GDPR-compliant."
    elif risk < 0.5:
        return "Dataset moderately safe; minor compliance improvements suggested."
    else:
        return "Dataset exhibits high privacy risk — additional anonymization recommended."


def print_report_summary(report):
    """
    Prints a concise human-readable summary to console.
    """
    print("\n=== PRIVACY VALIDATION SUMMARY ===")
    summary = report.get("summary", {})
    profiling = summary.get("profiling", {})
    pes = summary.get("risk_assessment", {})
    techniques = summary.get("decision_making", [])
    compliance = summary.get("compliance", {})
    insights = report.get("insights", {})

    print(f"🕒 Report Generated: {report['report_generated_at']}")
    print("\n🔹 Profiling Metrics:")
    for k, v in profiling.items():
        print(f"   - {k}: {v}")

    print(f"\n🔹 Privacy Exposure Score (PES): {pes.get('PES', 'N/A')} → Risk Level: {pes.get('risk_level', 'N/A')}")
    print("\n🔹 Techniques Applied:")
    for d in techniques:
        print(f"   - {d['technique']}: {d['reason']}")

    print("\n🔹 Compliance Summary:")
    for k, v in compliance.items():
        print(f"   - {k}: {v}")

    print("\n💡 Insights:")
    for k, v in insights.items():
        print(f"   - {k}: {v}")

    print("==================================\n")
