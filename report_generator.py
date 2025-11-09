# report_generator.py — Unified Logging and Closed-Loop Report Synthesis
# ----------------------------------------------------------------------
# Collects intermediate results from each stage and compares pre/post
# transformation privacy metrics for end-to-end validation.

import json
import os
from datetime import datetime

LOG_FILE = "logs/system_log.json"
FINAL_REPORT = "logs/final_report.json"


def log_step(step_name, payload):
    """
    Existing logging hook (keeps pipeline backwards compatible).
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().isoformat()

    entry = {
        "step": step_name,
        "timestamp": timestamp,
        "data": payload
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


def generate_final_report(before_pes, after_pes, initial_utility=None, final_utility=None, out_path="logs/final_report.json"):
    """
    Generate final report and persist to disk. Added optional initial_utility
    and final_utility fields so the report includes usefulness metrics.
    """
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "before_pes": before_pes,
        "after_pes": after_pes,
        "initial_utility": initial_utility,
        "final_utility": final_utility
    }

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except Exception as e:
        # best-effort: print warning but do not crash pipeline
        print(f"[WARNING] Could not write final report to {out_path}: {e}")

    # keep the existing logging hook behavior if present
    try:
        log_step("final_report", report)
    except Exception:
        pass

    return report
