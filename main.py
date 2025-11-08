# pipeline orchestrator

# Coordinates the sequential execution of all modules.
# main.py — Pipeline Orchestrator

import pandas as pd
import json
from profile import (
    uniqueness_ratio,
    entropy,
    mutual_info,
    kl_divergence,
    outlier_index,
    dimensionality
)
from risk_assessment import calculate_pes
from decision_making_engine import decide_techniques
from techniques import (
    k_anonymity, l_diversity, t_closeness, generalisation, suppression,
    permutation, differential_privacy, top_bottom_coding, bucketization
)
from compliance import perform_compliance_check
from report_generator import log_step, generate_final_report


class PrivacyFramework:
    """A framework for privacy-preserving data processing."""
    
    def __init__(self):
        """Initialize the privacy framework."""
        self.dataset = None
        self.metadata = None
        self.profile_metrics = None
        self.transformed_data = None
        
        self.available_techniques = {
            "k_anonymity": k_anonymity,
            "l_diversity": l_diversity,
            "t_closeness": t_closeness,
            "generalisation": generalisation,
            "suppression": suppression,
            "permutation": permutation,
            "differential_privacy": differential_privacy,
            "top_bottom_coding": top_bottom_coding,
            "bucketization": bucketization
        }

    def load_dataset(self, path):
        """Load dataset (expects CSV for now)."""
        try:
            self.dataset = pd.read_csv(path)
            print(f"[INFO] Loaded dataset: {path} ({self.dataset.shape[0]} rows, {self.dataset.shape[1]} cols)")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def load_metadata(self, path):
        """Load dataset metadata (JSON)."""
        try:
            with open(path, "r") as f:
                self.metadata = json.load(f)
            print(f"[INFO] Loaded metadata from {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading metadata: {e}")

    def process(self, dataset_path, metadata_path):
        # === 1. Load Inputs ===
        self.load_dataset(dataset_path)
        self.load_metadata(metadata_path)

        # === 2. Profiling ===
        self.profile_metrics = {
            "uniqueness_ratio": uniqueness_ratio(self.dataset),
            "entropy": entropy(self.dataset),
            "mutual_info": mutual_info(self.dataset),
            "kl_divergence": kl_divergence(self.dataset),
            "outlier_index": outlier_index(self.dataset),
            "dimensionality": dimensionality(self.dataset)
        }
        log_step("profiling", self.profile_metrics)

        # === 3. Risk Assessment ===
        pes_data = calculate_pes(self.profile_metrics)
        log_step("risk_assessment", pes_data)

        # === 4. Decision-Making ===
        decisions = decide_techniques(self.profile_metrics, pes_data["PES"])
        log_step("decision_making", decisions)

        # === 5. Apply Techniques ===
        self.transformed_data = self.dataset.copy()

        for d in decisions:
            tech = d["technique"]
            if tech in self.available_techniques:
                print(f"[INFO] Applying {tech}...")
                self.transformed_data = self.available_techniques[tech](self.transformed_data)
            else:
                print(f"[WARNING] Technique {tech} not implemented yet.")
        log_step("techniques_applied", [d["technique"] for d in decisions])

        # === 6. Compliance Check ===
        compliance_report = perform_compliance_check(self.transformed_data, self.metadata)
        log_step("compliance", compliance_report)

        # === 7. Generate Final Report ===
        generate_final_report()
        
        return self.transformed_data


if __name__ == "__main__":
    # Example usage
    framework = PrivacyFramework()
    transformed_data = framework.process("path/to/dataset.csv", "path/to/metadata.json")

    print("\n✅ Privacy-Preserving Validation Pipeline Completed.")


if __name__ == "__main__":
    # Example placeholders — you’ll replace with actual test dataset paths later
    dataset_path = "datasets/sample_dataset.csv"
    metadata_path = "datasets/sample_metadata.json"

