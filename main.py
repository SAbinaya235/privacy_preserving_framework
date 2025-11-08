# pipeline orchestrator

# Coordinates the sequential execution of all modules.

from profile import *
from risk_assessment import calculate_pes
from decision_making_engine import decide_techniques
from techniques import *
from compliance import perform_compliance_check
from report_generator import log_step, generate_final_report

def main(dataset, metadata):
    # 1. Profile dataset
    profile_metrics = {
        "uniqueness_ratio": uniqueness_ratio(dataset),
        "entropy": entropy(dataset),
        "mutual_info": mutual_info(dataset),
        "kl_divergence": kl_divergence(dataset),
        "outlier_index": outlier_index(dataset),
        "dimensionality": dimensionality(dataset)
    }
    log_step("profiling", profile_metrics)

    # 2. Calculate PES
    pes_data = calculate_pes(profile_metrics)
    log_step("risk_assessment", pes_data)

    # 3. Decide Techniques
    decisions = decide_techniques(profile_metrics, pes_data["PES"])
    log_step("decision_making", decisions)

    # 4. Apply Techniques (placeholder: sequentially apply)
    transformed_data = dataset
    for d in decisions:
        tech_fn = globals()[d["technique"]]
        transformed_data = tech_fn(transformed_data)
    log_step("techniques_applied", [d["technique"] for d in decisions])

    # 5. Compliance Check
    compliance_report = perform_compliance_check(transformed_data, metadata)
    log_step("compliance", compliance_report)

    # 6. Final Report
    generate_final_report()

if __name__ == "__main__":
    main("path_to_dataset.csv", "metadata.json")
