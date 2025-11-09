# report_generator.py — Unified Logging and Closed-Loop Report Synthesis
# ----------------------------------------------------------------------
# Collects intermediate results from each stage and compares pre/post
# transformation privacy metrics for end-to-end validation.

import json
import os
import datetime

LOG_FILE = "logs/system_log.json"
FINAL_REPORT = "logs/final_report.json"


def log_step(step_name, data):
    """Enhanced logger – captures reasoning and evidence for applied techniques."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()

    entry = {
        "step": step_name,
        "timestamp": timestamp,
        "data": data
    }

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump({"log_entries": []}, f, indent=4)

    with open(LOG_FILE, "r+") as f:
        log_data = json.load(f)
        log_data["log_entries"].append(entry)
        f.seek(0)
        json.dump(log_data, f, indent=4)
        f.truncate()


def generate_final_report(before_pes=None, after_pes=None):
    """
    Generate a closed-loop privacy validation report.
    Includes before/after metrics and privacy improvement evaluation.
    """
    if not os.path.exists(LOG_FILE):
        print("No logs found to generate a report.")
        return

    with open(LOG_FILE, "r") as f:
        log_data = json.load(f)

    report = {
        "report_generated_at": datetime.datetime.now().isoformat(),
        "summary": {},
        "detailed_log": log_data["log_entries"],
        "pre_post_comparison": {}
    }

    # Extract summarized data
    for entry in log_data["log_entries"]:
        step = entry["step"]
        report["summary"][step] = entry["data"]

    # ✅ Include pre/post PES comparison if available
    if before_pes and after_pes:
        report["pre_post_comparison"] = _compare_pes(before_pes, after_pes)

    # Attach additional compliance and risk interpretation
    _attach_insights(report, before_pes, after_pes)

    # Save to file
    os.makedirs("logs", exist_ok=True)
    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Final Privacy Validation Report Generated Successfully!")
    print(f"📄 Location: {FINAL_REPORT}")
    print_report_summary(report)


def _compare_pes(before, after):
    """Compare pre/post PES values and derive improvement metrics."""
    before_value = before.get("PES") or before.get("score", 0)
    after_value = after.get("PES") or after.get("score", 0)

    improvement = before_value - after_value
    improvement_pct = (improvement / before_value * 100) if before_value else 0

    return {
        "before_PES": round(before_value, 4),
        "after_PES": round(after_value, 4),
        "absolute_change": round(improvement, 4),
        "percentage_improvement": f"{improvement_pct:.2f}%",
        "status": "Improved" if improvement > 0 else "Degraded" if improvement < 0 else "No Change"
    }


def _attach_insights(report, before_pes, after_pes):
    """Adds compliance and risk interpretation summary."""
    summary = report["summary"]
    compliance_score = summary.get("compliance_results", {}).get("compliance_score", None)

    pre_post = report.get("pre_post_comparison", {})
    risk_reduction = pre_post.get("absolute_change", 0)
    risk_trend = pre_post.get("status", "N/A")

    report["insights"] = {
        "initial_privacy_exposure": pre_post.get("before_PES", None),
        "final_privacy_exposure": pre_post.get("after_PES", None),
        "risk_reduction": f"{risk_reduction:.3f}",
        "trend": risk_trend,
        "overall_compliance_score": compliance_score,
        "remarks": _generate_remark(pre_post, compliance_score)
    }


def _generate_remark(pre_post, compliance_score):
    """Generate contextual remark based on improvements and compliance."""
    risk_reduction = pre_post.get("absolute_change", 0)
    trend = pre_post.get("status", "N/A")

    if trend == "Improved" and compliance_score and compliance_score >= 0.8:
        return "Dataset shows improved privacy and strong compliance — ready for deployment."
    elif trend == "Improved" and compliance_score:
        return "Privacy improved, but compliance needs attention."
    elif trend == "No Change":
        return "No effective change detected — review anonymization parameters."
    else:
        return "Privacy degradation detected — re-evaluate chosen techniques."


def print_report_summary(report):
    """Prints a concise human-readable summary to console."""
    print("\n=== PRIVACY VALIDATION SUMMARY ===")
    summary = report.get("summary", {})
    profiling = summary.get("profiling_metrics", {})
    pre_post = report.get("pre_post_comparison", {})
    compliance = summary.get("compliance_results", {})
    insights = report.get("insights", {})

    print(f"🕒 Report Generated: {report['report_generated_at']}")

    print("\n🔹 Profiling Metrics (Pre):")
    for k, v in profiling.items():
        print(f"   - {k}: {v}")

    print("\n🔹 Privacy Exposure Comparison:")
    print(f"   - Before PES: {pre_post.get('before_PES', 'N/A')}")
    print(f"   - After PES:  {pre_post.get('after_PES', 'N/A')}")
    print(f"   - Δ Change:   {pre_post.get('absolute_change', 'N/A')} "
          f"({pre_post.get('percentage_improvement', 'N/A')}) → {pre_post.get('status', 'N/A')}")

    print("\n🔹 Compliance Summary:")
    if isinstance(compliance, dict):
        for k, v in compliance.items():
            print(f"   - {k}: {v}")
    else:
        print("   - No compliance data available.")

    print("\n💡 Insights:")
    for k, v in insights.items():
        print(f"   - {k}: {v}")

    print("==================================\n")
