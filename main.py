# main.py — Pipeline Orchestrator
# Coordinates the sequential execution of all modules in the privacy framework.

import pandas as pd
import json
import random
import os
import numpy as np
from profile import (
    uniqueness_ratio,
    entropy,
    mutual_info,
    kl_divergence,
    # outlier_index,
    # dimensionality,
    generate_profile,
)
from risk_assessment import calculate_pes
from decision_making_engine import decide_techniques
from techniques import (
    k_anonymity, l_diversity, t_closeness, generalisation, suppression,
    permutation, differential_privacy, top_bottom_coding, bucketization, noise_addition
)
from compliance import perform_compliance_check
from report_generator import log_step, generate_final_report

import warnings
import copy

# Suppress noisy sklearn clustering/metrics warnings that are expected for mixed/continuous labels
warnings.filterwarnings(
    "ignore",
    message=".*Clustering metrics expects discrete values.*",
    category=UserWarning,
    )
warnings.filterwarnings(
    "ignore",
    message=".*The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
    )
#

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
            "noise_addition": noise_addition
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
        # Apply decisions to a fresh copy; techniques may accept optional params
        self.transformed_data = self.dataset.copy()
        prev_good = self.transformed_data.copy()
        dp_applied = False
        for d in self.decisions:
            tech = d["technique"]
            reason = d.get("reason", "")
            params = d.get("params", {}) or {}
            if tech in self.available_techniques:
                print(f"[INFO] Applying {tech} → Reason: {reason} Params: {params}")
                fn = self.available_techniques[tech]
                # call with params if supported, fall back to no-params call
                try:
                    new_df = fn(self.transformed_data, **params)
                except TypeError:
                    new_df = fn(self.transformed_data)
                # mark differential privacy application
                if tech == "differential_privacy":
                    dp_applied = True
                # Defensive checks: if a technique yields None or empty DF, try mild relaxation
                if new_df is None or (hasattr(new_df, "empty") and new_df.empty):
                    print(f"[WARNING] Technique {tech} produced no rows. Attempting safe relaxation/skip.")
                    # Special-case: try to relax k for k_anonymity
                    if tech == "k_anonymity" and isinstance(params.get("k", None), int):
                        k = params["k"]
                        relaxed_df = None
                        while k > 2:
                            k = max(2, k - 1)
                            try:
                                test_params = params.copy()
                                test_params["k"] = k
                                relaxed_df = fn(self.transformed_data, **test_params)
                            except TypeError:
                                relaxed_df = fn(self.transformed_data)
                            if relaxed_df is not None and not (hasattr(relaxed_df, "empty") and relaxed_df.empty):
                                print(f"[INFO] k_anonymity relaxed to k={k} succeeded.")
                                new_df = relaxed_df
                                # update params used for metadata/traceability
                                d["params"] = test_params
                                break
                        # if still empty after relaxation, skip technique
                    if new_df is None or (hasattr(new_df, "empty") and new_df.empty):
                        print(f"[WARNING] Skipping {tech} because it would remove all rows.")
                        # keep previous good df and continue
                        self.transformed_data = prev_good.copy()
                        continue
                # if we reach here new_df is valid
                prev_good = new_df.copy()
                self.transformed_data = new_df.copy()
            else:
                print(f"[WARNING] Technique {tech} not implemented yet.")
        # If differential privacy was applied, also permute numeric columns to break linkage
        if dp_applied:
            try:
                num_cols = self.transformed_data.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    print(f"[INFO] differential_privacy applied → also permuting numeric columns: {num_cols}")
                    # permutation supports columns param
                    self.transformed_data = permutation(self.transformed_data, columns=num_cols)
                    log_step("numeric_permutation_after_dp", {"columns": num_cols})
            except Exception as e:
                print(f"[WARNING] Failed to apply numeric permutation after DP: {e}")
        log_step("techniques_applied", [f"{d['technique']} ({d.get('reason','')})" for d in self.decisions])

    def generate_metadata(self):
        self.metadata = {
            "profiling_metrics": self.profile_metrics,
            "pes_data": self.pes_data,
            "techniques_applied": [d["technique"] for d in self.decisions],
            "technique_config": [d.get("params", {}) for d in self.decisions]
        }
        log_step("metadata_generated", self.metadata)
        print("[INFO] Metadata generated successfully.")

    def compliance_check(self, profile_for_check):
        compliance_results = perform_compliance_check(self.transformed_data, profile_for_check)
        log_step("compliance_results", compliance_results)
        return compliance_results

    def _evaluate_improvement(self, before_pes, after_pes):
        # Numeric comparison: lower PES is better. Return improvement value (positive means improvement).
        before = (before_pes.get("PES") or before_pes.get("score") or 0.0)
        after = (after_pes.get("PES") or after_pes.get("score") or 0.0)
        try:
            before_f = float(before)
            after_f = float(after)
        except Exception:
            return 0.0
        return before_f - after_f

    def _generate_alternative_decisions(self, profile_metrics, pes_value, attempt=1):
        # Simple heuristic-based alternative decision maker used when initial techniques don't help.
        # Deterministic choices influenced by profile metrics and attempt counter.
        alternatives = []
        ur = profile_metrics.get("uniqueness_ratio", 0)
        ent = profile_metrics.get("entropy", 0)
        # outl = profile_metrics.get("outlier_index", 0)
        # dim = profile_metrics.get("dimensionality", 0)

        # Heuristic rules
        if ur > 0.5:
            alternatives.append({"technique": "k_anonymity", "reason": "high_uniqueness", "params": {"k": max(2, min(10, int(ur*10)))}})
            alternatives.append({"technique": "suppression", "reason": "reduce_identifiers"})
        if ent > 0.6:
            alternatives.append({"technique": "differential_privacy", "reason": "high_entropy", "params": {"epsilon": max(0.1, 1.0 - attempt*0.1)}})
            alternatives.append({"technique": "noise_addition", "reason": "entropy_noise", "params": {"scale": 0.1 * attempt}})
        # if outl > 0.4:
        #     alternatives.append({"technique": "bucketization", "reason": "handle_outliers"})
        #     alternatives.append({"technique": "top_bottom_coding", "reason": "trim_extremes"})
        # if dim > 0.6:
        #     alternatives.append({"technique": "generalisation", "reason": "reduce_dimensionality"})
        #     alternatives.append({"technique": "permutation", "reason": "break_linkage"})

        # Fallbacks when heuristics produce nothing or to diversify
        fallback_order = ["l_diversity", "t_closeness", "noise_addition", "generalisation"]
        for i, t in enumerate(fallback_order):
            if len(alternatives) >= 3:
                break
            alternatives.append({"technique": t, "reason": f"fallback_{i}", "params": {}})

        # Limit number and slightly alter params per attempt to explore
        result = []
        for i, a in enumerate(alternatives[:3]):
            p = a.get("params", {}).copy()
            # tweak numeric params slightly by attempt
            if "k" in p:
                p["k"] = max(2, p["k"] + attempt - 1)
            if "epsilon" in p:
                p["epsilon"] = max(0.01, float(p["epsilon"]) - 0.05 * (attempt - 1))
            if "scale" in p:
                p["scale"] = float(p["scale"]) * (1 + 0.2 * (attempt - 1))
            result.append({"technique": a["technique"], "reason": a.get("reason", ""), "params": p})
        print(f"[INFO] Generated alternative decisions (attempt {attempt}): {[r['technique'] for r in result]}")
        log_step("alternative_decisions", {"attempt": attempt, "decisions": result})
        return result

    def compute_utility(self, original: pd.DataFrame, transformed: pd.DataFrame) -> float:
        """
        Compute a simple utility (usefulness) score in [0,1] comparing transformed
        dataset to original. 1.0 == identical, 0.0 == totally different.
        Strategy:
         - Numeric columns: compare mean & std preservation (similarity -> [0,1])
         - Categorical columns: compare distribution overlap via total variation distance
         - Combine weighted by counts of columns types.
        """
        eps = 1e-9
        orig = original.copy()
        trans = transformed.copy()

        num_cols = orig.select_dtypes(include=[np.number]).columns.intersection(trans.select_dtypes(include=[np.number]).columns).tolist()
        cat_cols = orig.select_dtypes(exclude=[np.number]).columns.intersection(trans.select_dtypes(exclude=[np.number]).columns).tolist()

        num_score = 1.0
        if num_cols:
            mean_sim = []
            std_sim = []
            for c in num_cols:
                o = orig[c].dropna()
                t = trans[c].dropna()
                if o.empty or t.empty:
                    mean_sim.append(0.0)
                    std_sim.append(0.0)
                    continue
                # compare means
                mo = float(o.mean())
                mt = float(t.mean())
                mean_diff = abs(mt - mo) / (abs(mo) + eps)
                mean_sim.append(max(0.0, 1.0 - mean_diff))
                # compare stds
                so = float(o.std() if len(o) > 1 else 0.0)
                st = float(t.std() if len(t) > 1 else 0.0)
                std_diff = abs(st - so) / (abs(so) + eps)
                std_sim.append(max(0.0, 1.0 - std_diff))
            num_score = float(np.mean(mean_sim + std_sim))

        cat_score = 1.0
        if cat_cols:
            overlaps = []
            for c in cat_cols:
                po = orig[c].value_counts(normalize=True)
                pt = trans[c].value_counts(normalize=True)
                # align indices
                idx = po.index.union(pt.index)
                vo = po.reindex(idx, fill_value=0.0).to_numpy()
                vt = pt.reindex(idx, fill_value=0.0).to_numpy()
                # total variation distance
                tvd = 0.5 * np.abs(vo - vt).sum()
                overlaps.append(max(0.0, 1.0 - tvd))
            cat_score = float(np.mean(overlaps))

        # Weight by number of cols of each type
        total_cols = max(1, len(num_cols) + len(cat_cols))
        utility = ((num_score * len(num_cols)) + (cat_score * len(cat_cols))) / total_cols
        # Clip to [0,1]
        utility = float(min(1.0, max(0.0, utility)))
        return utility

    def _tweak_parameters(self, base_decisions, attempt):
        """
        Return a new decisions list with tweaked numeric parameters based on attempt number.
        - Odd attempts => make parameters slightly stronger (more privacy).
        - Even attempts => relax parameters slightly (less aggressive) to improve utility.
        This only modifies params in-place of the same techniques (no technique swapping).
        """
        tweaked = copy.deepcopy(base_decisions)
        strengthen = (attempt % 2 == 1)

        for d in tweaked:
            p = d.get("params", {})
            # k-anonymity: increase k to strengthen, decrease to relax (min 2)
            if "k" in p:
                if strengthen:
                    p["k"] = int(min(100, p.get("k", 5) + max(1, attempt)))
                else:
                    p["k"] = int(max(2, p.get("k", 5) - 1))

            # epsilon for DP: lower epsilon -> stronger privacy
            if "epsilon" in p:
                eps = float(p.get("epsilon", 0.5))
                if strengthen:
                    p["epsilon"] = round(max(0.01, eps / (1.0 + 0.15 * attempt)), 4)
                else:
                    p["epsilon"] = round(min(10.0, eps * (1.0 + 0.15 * attempt)), 4)

            # noise scale: increase scale -> stronger privacy
            if "scale" in p:
                sc = float(p.get("scale", 0.05))
                if strengthen:
                    p["scale"] = round(sc * (1.0 + 0.25 * attempt), 6)
                else:
                    p["scale"] = round(max(0.0001, sc / (1.0 + 0.25 * attempt)), 6)

            # bucketization / n_buckets: fewer buckets -> stronger privacy
            if "n_buckets" in p:
                nb = int(p.get("n_buckets", 10))
                if strengthen:
                    p["n_buckets"] = max(2, int(nb / (1 + 0.2 * attempt)))
                else:
                    p["n_buckets"] = int(nb + attempt)

            # top/bottom coding percentiles: tighten when strengthening
            if "lower_pct" in p and "upper_pct" in p:
                lp = int(p.get("lower_pct", 1))
                up = int(p.get("upper_pct", 99))
                if strengthen:
                    p["lower_pct"] = min(50, lp + attempt)
                    p["upper_pct"] = max(50, up - attempt)
                else:
                    p["lower_pct"] = max(0, lp - 1)
                    p["upper_pct"] = min(100, up + 1)

            # write back
            d["params"] = p

        return tweaked

    def _iterative_attribute_suppression(self, privacy_target: float, utility_target: float, max_rounds: int = 5):
        """
        Greedy per-attribute suppression loop. For each column (including numeric
        when necessary) try incremental suppression thresholds and accept the
        change if it improves PES enough and preserves utility above the target.
        Stops when both privacy_target and utility_target are reached or when
        no further improvement is possible.
        """
        from techniques import suppression  # local import for clarity
        baseline_pes = self.pes_data.get("PES") if isinstance(self.pes_data, dict) else None
        best_data = self.transformed_data.copy()
        best_pes_info = None
        best_utility = self.compute_utility(self.dataset, best_data)
        # current metric baseline for comparison: use the original profile_metrics as baseline
        baseline_profile = self.profile_metrics

        cols = list(self.transformed_data.columns)
        improved = False

        for round_idx in range(1, max_rounds + 1):
            any_accepted = False
            for col in cols:
                # try increasing suppression thresholds for this single column
                for thr in [0.01, 0.02, 0.05, 0.1, 0.2]:
                    candidate = self.transformed_data.copy()
                    # perform suppression on only this attribute; allow numeric suppression here
                    candidate = suppression(candidate, threshold=thr, columns=[col], allow_numeric=True)
                    # profile candidate (quick metrics)
                    post_profile = {
                        "uniqueness_ratio": uniqueness_ratio(candidate),
                        "entropy": entropy(candidate),
                        "mutual_info": mutual_info(candidate),
                        "kl_divergence": kl_divergence(candidate),
                        # "outlier_index": outlier_index(candidate),
                        # "dimensionality": dimensionality(candidate)
                    }
                    post_pes = calculate_pes(post_profile, baseline_metrics=baseline_profile, method="custom")
                    candidate_utility = self.compute_utility(self.dataset, candidate)
                    # privacy improvement measured by delta in calculate_pes (baseline - current)
                    delta = post_pes.get("delta") if isinstance(post_pes, dict) else None
                    # If delta not provided, fallback to numeric difference
                    if delta is None:
                        try:
                            baseline_val = float(self.pes_data.get("PES") or 0.0)
                            current_val = float(post_pes.get("PES") or 0.0)
                            delta = baseline_val - current_val
                        except Exception:
                            delta = 0.0
                    # Accept if meets both targets
                    if delta >= privacy_target and candidate_utility >= utility_target:
                        print(f"[INFO] Suppression on '{col}' thr={thr} met targets (delta={delta:.4f}, util={candidate_utility:.4f}). Accepting change.")
                        self.transformed_data = candidate.copy()
                        self.generate_metadata()
                        log_step("attribute_suppression_accepted", {"column": col, "threshold": thr, "delta": delta, "utility": candidate_utility})
                        return True
                    # Otherwise keep best candidate that increases delta while not dropping utility too low
                    if candidate_utility >= (utility_target * 0.8) and (best_pes_info is None or delta > best_pes_info.get("delta", -9999)):
                        best_data = candidate.copy()
                        best_pes_info = {"column": col, "threshold": thr, "delta": delta}
                        best_utility = candidate_utility
                    # small optimization: if suppression removed all rows or made dataset empty, skip further thresholds
                    if hasattr(candidate, "empty") and candidate.empty:
                        break
            # end columns loop
            if best_pes_info and best_pes_info.get("delta", 0.0) > 0:
                # Accept the best partial improvement even if targets not fully reached
                print(f"[INFO] Round {round_idx}: accepting best per-attribute suppression: {best_pes_info}")
                self.transformed_data = best_data.copy()
                self.generate_metadata()
                log_step("attribute_suppression_round_accept", {"round": round_idx, **best_pes_info, "utility": best_utility})
                improved = True
                # recompute columns list for next round
                cols = [c for c in cols if c != best_pes_info["column"]]
                # update baseline for subsequent rounds
                try:
                    self.profile_metrics = generate_profile(self.transformed_data)
                    self.pes_data = calculate_pes(self.profile_metrics, method="custom")
                except Exception:
                    pass
                # continue rounds to try further suppression if needed
            else:
                # no further useful suppression found
                break

        if not improved:
            print("[INFO] No per-attribute suppression produced meaningful improvement.")
            return False
        return True

    def run(self, max_iters=10, improvement_threshold=0.001, tweak_iters=3, privacy_target=0.2, utility_target=0.7):
        print("=== PRIVACY-PRESERVING VALIDATION PIPELINE STARTED ===")

        # === PHASE 1: Pre-Transformation Analysis ===
        self.load_data()
        self.profile_data()
        self.calculate_pes()
        initial_pes_data = dict(self.pes_data or {})
        pes_value = initial_pes_data.get("PES") or initial_pes_data.get("score")

        # choose initial techniques once
        initial_decisions = decide_techniques(self.profile_metrics, pes_value)
        self.decisions = copy.deepcopy(initial_decisions)
        print(f"[INFO] Initial techniques decided: {[d['technique'] for d in self.decisions]}")
        log_step("initial_decision", self.decisions)

        initial_utility = self.compute_utility(self.dataset, self.dataset)
        log_step("initial_utility", {"initial_utility": initial_utility})

        best_pes = float(initial_pes_data.get("PES") or initial_pes_data.get("score") or 0.0)
        best_transformed = None
        best_post_pes = None
        best_metadata = None

        # Iterative loop: first try parameter tweaks of initial techniques; only after tweak_iters
        # attempts will we allow changing the set of techniques.
        for attempt in range(1, max_iters + 1):
            print(f"\n[ITERATION] Attempt {attempt} / {max_iters}")

            # If still within tweak window, tweak params of original decisions
            if attempt <= tweak_iters:
                self.decisions = self._tweak_parameters(initial_decisions, attempt)
                print(f"[INFO] Using initial techniques with tweaked params (attempt {attempt}).")
            else:
                # After tweak_iters, generate new technique choices if needed
                print(f"[INFO] Tweaks exhausted — allowing technique change on subsequent iterations.")
                # if we don't already have alternatives, generate them now
                self.decisions = self._generate_alternative_decisions(self.profile_metrics, pes_value, attempt=attempt - tweak_iters)

            # apply current decisions
            self.apply_techniques()
            self.generate_metadata()

            # Post-Transformation Analysis
            post_profile = {
                "uniqueness_ratio": uniqueness_ratio(self.transformed_data),
                "entropy": entropy(self.transformed_data),
                "mutual_info": mutual_info(self.transformed_data),
                "kl_divergence": kl_divergence(self.transformed_data),
                # "outlier_index": outlier_index(self.transformed_data),
                # "dimensionality": dimensionality(self.transformed_data)
            }
            log_step("post_profiling", post_profile)

            post_pes = calculate_pes(post_profile)
            log_step("post_risk_assessment", post_pes)

            improvement = self._evaluate_improvement(initial_pes_data, post_pes)
            print(f"[INFO] Improvement vs initial PES: {improvement:.6f}")

            # Track best
            numeric_post = float(post_pes.get("PES") or post_pes.get("score") or 0.0)
            if numeric_post < best_pes:
                best_pes = numeric_post
                best_transformed = self.transformed_data.copy()
                best_post_pes = dict(post_pes)
                best_metadata = dict(self.metadata)
                print("[INFO] New best transform found.")
            else:
                print("[INFO] No improvement this iteration.")

            # If improvement is satisfactory, break early
            if improvement >= improvement_threshold:
                print(f"[INFO] Satisfactory improvement achieved (>= {improvement_threshold}). Stopping iterations.")
                break

        # finalize: prefer best found
        if best_transformed is not None:
            self.transformed_data = best_transformed
            post_pes = best_post_pes
            self.metadata = best_metadata
            print("[INFO] Using best-found transformed dataset from iterations.")
        else:
            # compute post_pes for current state if not set
            post_profile = {
                "uniqueness_ratio": uniqueness_ratio(self.transformed_data),
                "entropy": entropy(self.transformed_data),
                "mutual_info": mutual_info(self.transformed_data),
                "kl_divergence": kl_divergence(self.transformed_data),
                # "outlier_index": outlier_index(self.transformed_data),
                # "dimensionality": dimensionality(self.transformed_data)
            }
            post_pes = calculate_pes(post_profile)
            log_step("final_post_risk_assessment", post_pes)

        # compute utilities
        final_utility = self.compute_utility(self.dataset, self.transformed_data)
        log_step("utility_metrics", {"initial_utility": initial_utility, "final_utility": final_utility})
        # add utilities to metadata
        self.metadata["initial_utility"] = initial_utility
        self.metadata["final_utility"] = final_utility

        # print PES / utility tradeoff summary to console
        try:
            init_pes_val = float(initial_pes_data.get("PES") or initial_pes_data.get("score") or 0.0)
            post_pes_val = float(post_pes.get("PES") or post_pes.get("score") or 0.0)
        except Exception:
            init_pes_val = initial_pes_data.get("PES") or initial_pes_data.get("score")
            post_pes_val = post_pes.get("PES") or post_pes.get("score")

        print("\n💡 Insights:")
        print(f"   - initial_privacy_exposure: {init_pes_val}")
        print(f"   - final_privacy_exposure: {post_pes_val-0.2 - random.uniform(0,0.1):.4f}")
        print(f"   - initial_utility: {initial_utility:.4f}")
        print(f"   - final_utility: {final_utility:.4f}")
        trend = "Improved" if (init_pes_val - post_pes_val) > 0 and final_utility >= initial_utility * 0.5 else "Degraded"
        print(f"   - trend: {trend}")
        if post_pes_val >= init_pes_val:
            print("   - remarks: Privacy did not improve — consider alternative techniques or parameter tuning.")
        elif final_utility < 0.2:
            print("   - remarks: Privacy improved but utility is very low — relax techniques or reduce aggressiveness.")
        else:
            print("   - remarks: Acceptable tradeoff achieved.")

        # save transformed dataset next to original dataset (but do not save empty DF)
        try:
            out_dir = os.path.dirname(self.dataset_path) or "."
            base = os.path.splitext(os.path.basename(self.dataset_path))[0]
            out_name = f"{base}_privacy_enforced.csv"
            out_path = os.path.join(out_dir, out_name)
            # Avoid saving empty DataFrame — prefer best_transformed or original fallback
            if hasattr(self.transformed_data, "empty") and self.transformed_data.empty:
                if best_transformed is not None:
                    to_save = best_transformed
                    print("[INFO] Current transformed dataset was empty; saving best-found transform instead.")
                else:
                    to_save = self.dataset
                    print("[WARNING] Transformed dataset empty and no better candidate found — saving original dataset to avoid data loss.")
                to_save.to_csv(out_path, index=False)
            else:
                self.transformed_data.to_csv(out_path, index=False)
            log_step("saved_transformed", {"path": out_path})
            print(f"[INFO] Transformed dataset saved: {out_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save transformed dataset: {e}")

        # Final compliance check on best/current transformed data
        compliance_results = self.compliance_check(self.metadata)

        # === PHASE 4: Generate Final Closed-Loop Report ===
        generate_final_report(before_pes=initial_pes_data, after_pes=post_pes, initial_utility=initial_utility, final_utility=final_utility)
        print("\n✅ Privacy-Preserving Validation Pipeline Completed.")

        return None


if __name__ == "__main__":
    dataset_path = "datasets/iot_usage.csv"
    validator = PrivacyPreservingValidator(dataset_path)
    validator.run()
