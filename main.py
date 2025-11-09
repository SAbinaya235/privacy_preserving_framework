# main.py — Pipeline Orchestrator
# Coordinates the sequential execution of all modules in the privacy framework.

import pandas as pd
import json
from profile import (
    uniqueness_ratio,
    entropy,
    mutual_info,
    kl_divergence,
    outlier_index,
    dimensionality,
    generate_profile,
)
from risk_assessment import calculate_pes
from decision_making_engine import decide_techniques
from techniques import (
    k_anonymity, l_diversity, t_closeness, generalisation, suppression,
    permutation, differential_privacy, top_bottom_coding, bucketization
)
from compliance import perform_compliance_check
from report_generator import log_step, generate_final_report


class PrivacyPreservingValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.profile_metrics = None
        self.pes_data = None
        self.decisions = None
        self.transformed_data = None
        self.metadata = {}
        self.available_techniques = {
            "k_anonymity": k_anonymity,
            "l_diversity": l_diversity,
            "t_closeness": t_closeness,
            "generalisation": generalisation,
            "suppression": suppression,
            "permutation": permutation,
            "differential_privacy": differential_privacy,
            "top_bottom_coding": top_bottom_coding,
            "bucketization": bucketization,
        }

    def load_dataset(self, path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {path}: {e}")

    def load_data(self):
        self.dataset = self.load_dataset(self.dataset_path)
        print(f"[INFO] Loaded dataset: {self.dataset_path} "
              f"({len(self.dataset)} rows, {len(self.dataset.columns)} cols)")
        log_step("dataset_loaded", {"path": self.dataset_path})

    def profile_data(self):
        self.profile_metrics = generate_profile(self.dataset)
        log_step("profiling_metrics", self.profile_metrics)

    def calculate_pes(self):
        self.pes_data = calculate_pes(self.profile_metrics)
        pes_value = self.pes_data.get("PES") or self.pes_data.get("score")
        risk_level = self.pes_data.get("risk_level")
        if pes_value is not None:
            print(f"[INFO] Privacy Exposure Score (PES): {pes_value:.3f} → Risk Level: {risk_level}")
        else:
            print(f"[INFO] PES computed: {self.pes_data}")
        log_step("pes_score", self.pes_data)

    def make_decisions(self):
        pes_value = self.pes_data.get("PES") or self.pes_data.get("score")
        self.decisions = decide_techniques(self.profile_metrics, pes_value)
        print(f"[INFO] Techniques decided: {[d['technique'] for d in self.decisions]}")
        log_step("technique_decisions", self.decisions)

    def apply_techniques(self):
        self.transformed_data = self.dataset.copy()
        for d in self.decisions:
            tech = d["technique"]
            reason = d.get("reason", "")
            if tech in self.available_techniques:
                print(f"[INFO] Applying {tech} → Reason: {reason}")
                self.transformed_data = self.available_techniques[tech](self.transformed_data)
            else:
                print(f"[WARNING] Technique {tech} not implemented yet.")
        log_step("techniques_applied", [f"{d['technique']} ({d['reason']})" for d in self.decisions])

    def generate_metadata(self):
        self.metadata = {
            "profiling_metrics": self.profile_metrics,
            "pes_data": self.pes_data,
            "techniques_applied": [d["technique"] for d in self.decisions]
        }
        log_step("metadata_generated", self.metadata)
        print("[INFO] Metadata generated successfully.")

    def compliance_check(self, profile_for_check):
        compliance_results = perform_compliance_check(self.transformed_data, profile_for_check)
        log_step("compliance_results", compliance_results)
        return compliance_results

    def run(self):
        print("=== PRIVACY-PRESERVING VALIDATION PIPELINE STARTED ===")

        # === PHASE 1: Pre-Transformation Analysis ===
        self.load_data()
        self.profile_data()
        self.calculate_pes()
        self.make_decisions()

        # === PHASE 2: Apply Privacy Techniques ===
        self.apply_techniques()

        # === PHASE 3: Post-Transformation Analysis (Closed Loop) ===
        post_profile = {
            "uniqueness_ratio": uniqueness_ratio(self.transformed_data),
            "entropy": entropy(self.transformed_data),
            "mutual_info": mutual_info(self.transformed_data),
            "kl_divergence": kl_divergence(self.transformed_data),
            "outlier_index": outlier_index(self.transformed_data),
            "dimensionality": dimensionality(self.transformed_data)
        }
        log_step("post_profiling", post_profile)

        post_pes = calculate_pes(post_profile)
        log_step("post_risk_assessment", post_pes)

        compliance_results = self.compliance_check(post_profile)

        # === PHASE 4: Generate Final Closed-Loop Report ===
        generate_final_report(before_pes=self.pes_data, after_pes=post_pes)
        print("\n✅ Privacy-Preserving Validation Pipeline Completed.")

        return None


if __name__ == "__main__":
    dataset_path = "datasets/employee_records.csv"
    validator = PrivacyPreservingValidator(dataset_path)
    validator.run()
