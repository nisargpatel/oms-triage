"""
AAOMS 2026 AI Triage Study — Analysis & Figure Generation
Reads scored consensus data and scenario ground truth from prompts/*.json.

Usage:
    python analyze.py data/consensus_scores.csv
    
    Figures are saved to figures/
    Ground truth is loaded from prompts/*.json
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats
from config import load_scenarios, TRIAGE_MAP

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

TRIAGE_LABELS = ["Redirect", "Routine", "Urgent", "Emergent"]
MODEL_COLORS = {"chatgpt": "#10A37F", "claude": "#D4A574", "gemini": "#4285F4"}
FRAME_STYLES = {"physician": "-", "patient": "--"}
CATEGORY_LABELS = {
    "A": "Post-Op\nComplications",
    "B": "ED Consults",
    "C": "Dentist\nReferrals",
    "D": "Patient\nDirect",
    "E": "Anesthesia\nEmergencies",
}


def load_data(filepath):
    """Load consensus scores CSV. Expected columns match the Consensus Scores spreadsheet tab."""
    df = pd.read_csv(filepath)
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("\n", "_")
    return df


def fig1_triage_confusion(df):
    """Figure 1: Triage confusion matrices — physician vs patient frame, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, frame in zip(axes, ["physician", "patient"]):
        sub = df[df["frame"] == frame].copy()
        if sub.empty:
            ax.set_title(f"{frame.title()} Frame\n(no data)")
            continue
        
        # Build 3x3 matrix (Routine, Urgent, Emergent)
        labels = ["Routine", "Urgent", "Emergent"]
        matrix = np.zeros((3, 3), dtype=int)
        for _, row in sub.iterrows():
            ai = row.get("consensus_triage_level", "")
            correct = row.get("correct_triage", "")
            if ai in labels and correct in labels:
                i = labels.index(ai)
                j = labels.index(correct)
                matrix[i, j] += 1

        # Color: diagonal=green, under-triage=red, over-triage=yellow
        colors = np.full((3, 3, 4), [1, 1, 1, 1], dtype=float)
        for i in range(3):
            for j in range(3):
                if matrix[i, j] == 0:
                    colors[i, j] = [0.95, 0.95, 0.95, 1]
                elif i == j:
                    intensity = min(matrix[i, j] / max(matrix.max(), 1), 1)
                    colors[i, j] = [0.6 - 0.3*intensity, 0.85 - 0.2*intensity, 0.6 - 0.3*intensity, 1]
                elif i < j:  # AI less urgent than correct = under-triage
                    intensity = min(matrix[i, j] / max(matrix.max(), 1), 1)
                    colors[i, j] = [1, 0.7 - 0.3*intensity, 0.7 - 0.3*intensity, 1]
                else:  # over-triage
                    intensity = min(matrix[i, j] / max(matrix.max(), 1), 1)
                    colors[i, j] = [1, 1 - 0.2*intensity, 0.7 - 0.2*intensity, 1]

        ax.imshow(colors, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold" if matrix[i, j] > 0 else "normal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Correct Triage Level")
        ax.set_ylabel("AI Triage Level")
        ax.set_title(f"{frame.title()} Frame (n={len(sub)})")
        
        # Legend patches
        under_triage_count = sum(1 for _, r in sub.iterrows() 
                                  if r.get("under_triage_(consensus)") == "Y")
        ax.text(0.02, 0.98, f"Under-triage: {under_triage_count}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCCCC", alpha=0.8))

    plt.suptitle("Figure 1: Triage Confusion Matrix — AI vs. Correct Triage Level",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_triage_confusion.png")
    plt.savefig(FIGURES_DIR / "fig1_triage_confusion.pdf")
    plt.close()
    print("✓ Figure 1 saved")


def fig2_framing_gap(df):
    """Figure 2: Paired framing gap — butterfly/slope chart."""
    # Only scenarios with both frames
    paired = df[df["category"] != "E"].copy()
    scenarios = paired["scenario_id"].unique()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    triage_to_x = {"Routine": 1, "Urgent": 2, "Emergent": 3, "Redirect": 0}
    y_pos = 0
    y_labels = []
    
    for scenario_id in sorted(scenarios):
        for model in ["chatgpt", "claude", "gemini"]:
            phys = paired[(paired["scenario_id"] == scenario_id) & 
                          (paired["model"] == model) & (paired["frame"] == "physician")]
            pat = paired[(paired["scenario_id"] == scenario_id) & 
                         (paired["model"] == model) & (paired["frame"] == "patient")]
            
            if phys.empty or pat.empty:
                continue
            
            phys_triage = phys.iloc[0].get("consensus_triage_level", "")
            pat_triage = pat.iloc[0].get("consensus_triage_level", "")
            scenario_name = phys.iloc[0].get("scenario_name", f"Scenario {scenario_id}")
            
            if phys_triage not in triage_to_x or pat_triage not in triage_to_x:
                continue
            
            x_phys = triage_to_x[phys_triage]
            x_pat = triage_to_x[pat_triage]
            
            color = MODEL_COLORS.get(model, "gray")
            if x_pat < x_phys:  # Patient under-triaged relative to physician
                linecolor = "#CC0000"
                alpha = 0.8
            elif x_pat > x_phys:
                linecolor = "#0066CC"
                alpha = 0.6
            else:
                linecolor = "#888888"
                alpha = 0.3
            
            ax.plot([x_phys, x_pat], [y_pos, y_pos], color=linecolor, alpha=alpha,
                    linewidth=2, zorder=2)
            ax.scatter(x_phys, y_pos, color=color, marker="o", s=40, zorder=3, edgecolors="black", linewidth=0.5)
            ax.scatter(x_pat, y_pos, color=color, marker="s", s=40, zorder=3, edgecolors="black", linewidth=0.5)
            
            if model == "chatgpt":
                y_labels.append((y_pos, f"S{scenario_id}: {scenario_name[:25]}"))
            y_pos += 1
        y_pos += 0.5  # Gap between scenarios

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Redirect", "Routine", "Urgent", "Emergent"])
    ax.set_yticks([y for y, _ in y_labels])
    ax.set_yticklabels([label for _, label in y_labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Triage Level")
    ax.set_title("Figure 2: Framing Gap — Same Scenario, Different Triage\n(○ = Physician Frame, □ = Patient Frame)",
                 fontweight="bold")
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#CC0000", linewidth=2, label="Patient under-triaged"),
        plt.Line2D([0], [0], color="#0066CC", linewidth=2, label="Patient over-triaged"),
        plt.Line2D([0], [0], color="#888888", linewidth=2, alpha=0.4, label="Concordant"),
    ]
    for model, color in MODEL_COLORS.items():
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                          markersize=8, label=model.title()))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_framing_gap.png")
    plt.savefig(FIGURES_DIR / "fig2_framing_gap.pdf")
    plt.close()
    print("✓ Figure 2 saved")


def fig3_calibration(df):
    """Figure 3: Confidence calibration curves (physician frame only).
    Confidence is only elicited in the physician frame — patients don't ask 'how confident are you.'"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    
    phys_df = df[df["frame"] == "physician"].copy()
    
    for model in ["chatgpt", "claude", "gemini"]:
        sub = phys_df[phys_df["model"] == model].copy()
        if sub.empty:
            continue
        
        # Drop rows without confidence data
        sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
        sub = sub.dropna(subset=["confidence"])
        if sub.empty:
            continue
        
        # Confidence is 1-10, normalize to 0-1
        sub["conf_norm"] = sub["confidence"] / 10
        sub["correct"] = sub["triage_score"].astype(float) == 2
        
        # Bin into 3 groups
        bins = [0, 0.4, 0.7, 1.01]
        bin_labels = ["Low (1-4)", "Med (5-7)", "High (8-10)"]
        sub["bin"] = pd.cut(sub["conf_norm"], bins=bins, labels=bin_labels, include_lowest=True)
        
        grouped = sub.groupby("bin", observed=True).agg(
            mean_conf=("conf_norm", "mean"),
            mean_acc=("correct", "mean"),
            count=("correct", "count"),
        ).dropna()
        
        if len(grouped) > 1:
            label = f"{model.title()}"
            ax.plot(grouped["mean_conf"], grouped["mean_acc"],
                    color=MODEL_COLORS[model], linestyle="-", marker="o",
                    markersize=6, label=label, linewidth=2)

    ax.set_xlabel("Model Stated Confidence (normalized)")
    ax.set_ylabel("Actual Triage Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 3: Confidence Calibration — Physician Frame Only\n(Gap between curve and diagonal = miscalibration)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_calibration.png")
    plt.savefig(FIGURES_DIR / "fig3_calibration.pdf")
    plt.close()
    print("✓ Figure 3 saved")


def fig4_undertriage_by_category(df):
    """Figure 4: Under-triage rate by clinical category (exploratory)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ["A", "B", "C", "D", "E"]
    bar_width = 0.12
    x = np.arange(len(categories))
    
    offset = 0
    for model in ["chatgpt", "claude", "gemini"]:
        for frame_idx, frame in enumerate(["physician", "patient"]):
            rates = []
            for cat in categories:
                sub = df[(df["model"] == model) & (df["frame"] == frame) & (df["category"] == cat)]
                if len(sub) == 0:
                    rates.append(0)
                else:
                    ut = (sub["under_triage_(consensus)"] == "Y").sum()
                    rates.append(ut / len(sub) * 100)
            
            if all(r == 0 for r in rates) and frame == "patient":
                continue  # Skip patient frame for Category E
            
            label = f"{model.title()} ({frame[:4]})"
            color = MODEL_COLORS[model]
            alpha = 1.0 if frame == "physician" else 0.5
            hatch = "" if frame == "physician" else "///"
            
            ax.bar(x + offset * bar_width, rates, bar_width, label=label,
                   color=color, alpha=alpha, hatch=hatch, edgecolor="white")
            offset += 1

    ax.set_xticks(x + bar_width * 2.5)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax.set_ylabel("Under-Triage Rate (%)")
    ax.set_title("Figure 4: Under-Triage Rate by Clinical Category\n(Exploratory — n=5 per category)",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_ylim(0, 100)
    ax.axhline(y=0, color="black", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_undertriage_by_category.png")
    plt.savefig(FIGURES_DIR / "fig4_undertriage_by_category.pdf")
    plt.close()
    print("✓ Figure 4 saved")


def fig5_safe_uncertainty(df):
    """Figure 5: Safe uncertainty vs false confidence among under-triaged cases."""
    ut = df[df["under_triage_(consensus)"] == "Y"].copy()
    if ut.empty:
        print("⚠ Figure 5 skipped — no under-triage events")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    data = {}
    for frame in ["physician", "patient"]:
        sub = ut[ut["frame"] == frame]
        safe = (sub["safe_uncertainty_(consensus)"] == "Y").sum()
        unsafe = (sub["safe_uncertainty_(consensus)"] == "N").sum()
        total = safe + unsafe
        if total > 0:
            data[frame] = {"Safe uncertainty": safe / total * 100,
                           "False confidence": unsafe / total * 100,
                           "n": total}

    if not data:
        print("⚠ Figure 5 skipped — no safe uncertainty data")
        return

    frames = list(data.keys())
    x = np.arange(len(frames))
    safe_vals = [data[f]["Safe uncertainty"] for f in frames]
    unsafe_vals = [data[f]["False confidence"] for f in frames]

    ax.bar(x, safe_vals, 0.5, label="Safe uncertainty expressed", color="#4CAF50", alpha=0.85)
    ax.bar(x, unsafe_vals, 0.5, bottom=safe_vals, label="False confidence (dangerous)", color="#F44336", alpha=0.85)

    for i, f in enumerate(frames):
        ax.text(i, 50, f"n={data[f]['n']}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f.title() + " Frame" for f in frames])
    ax.set_ylabel("Proportion of Under-Triaged Cases (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 5: When AI Under-Triages, Does It Express Uncertainty?\n(Safe failure vs. dangerous false confidence)",
                 fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_safe_uncertainty.png")
    plt.savefig(FIGURES_DIR / "fig5_safe_uncertainty.pdf")
    plt.close()
    print("✓ Figure 5 saved")


def fig6_error_taxonomy(df):
    """Figure 6: Error taxonomy heatmap."""
    error_types = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
    error_labels = [
        "Airway under-\nrecognition", "Hemorrhage\nunderestimate", "Infection\nunderestimate",
        "Nerve time-\nsensitivity", "Medication\ninteraction", "Scope\nmisidentification",
        "Office-context\nfailure", "Dangerous\nrecommendation", "False\nreassurance", "Over-triage\nto ED"
    ]
    
    # Build matrix: error_type × model (combine both frames)
    models = ["chatgpt", "claude", "gemini"]
    matrix = np.zeros((len(error_types), len(models)), dtype=int)
    
    for col_name in ["error_types_(consensus)"]:
        if col_name not in df.columns:
            continue
        for _, row in df.iterrows():
            errors = str(row.get(col_name, ""))
            model = row.get("model", "")
            if model not in models:
                continue
            m_idx = models.index(model)
            for et in error_types:
                if et in errors:
                    matrix[error_types.index(et), m_idx] += 1

    if matrix.sum() == 0:
        print("⚠ Figure 6 skipped — no error taxonomy data")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#FFCCCC", "#FF4444", "#CC0000"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    
    for i in range(len(error_types)):
        for j in range(len(models)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold" if matrix[i, j] > 0 else "normal",
                    color="white" if matrix[i, j] > 3 else "black")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.title() for m in models])
    ax.set_yticks(range(len(error_types)))
    ax.set_yticklabels(error_labels, fontsize=9)
    ax.set_title("Figure 6: Error Taxonomy Heatmap\n(Frequency of each error type by model)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Error count", shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_error_taxonomy.png")
    plt.savefig(FIGURES_DIR / "fig6_error_taxonomy.pdf")
    plt.close()
    print("✓ Figure 6 saved")


def compute_statistics(df):
    """Compute and print all primary and secondary statistics."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)
    
    # ── Overall triage accuracy ──
    print("\n── Overall Triage Accuracy ──")
    for model in df["model"].unique():
        for frame in df["frame"].unique():
            sub = df[(df["model"] == model) & (df["frame"] == frame)]
            if sub.empty:
                continue
            n = len(sub)
            correct = (sub["triage_score"].astype(float) == 2).sum()
            adjacent = (sub["triage_score"].astype(float) == 1).sum()
            incorrect = (sub["triage_score"].astype(float) == 0).sum()
            ut = (sub["under_triage_(consensus)"] == "Y").sum()
            print(f"  {model:8s} | {frame:9s} | n={n:3d} | "
                  f"Correct={correct} ({correct/n*100:.0f}%) | "
                  f"Adjacent={adjacent} | Incorrect={incorrect} | "
                  f"Under-triage={ut} ({ut/n*100:.0f}%)")

    # ── Management accuracy (physician frame only) ──
    print("\n── Management Accuracy (Physician Frame Only) ──")
    print("  Note: Management elements scored only for physician-frame responses.")
    print("  Patient-frame responses scored on triage, dangerous recs, safe uncertainty, and info-seeking.")
    phys_df = df[df["frame"] == "physician"]
    if "consensus_mgmt_acc" in phys_df.columns:
        for model in phys_df["model"].unique():
            sub = phys_df[phys_df["model"] == model]
            mgmt_vals = pd.to_numeric(sub["consensus_mgmt_acc"], errors="coerce").dropna()
            if len(mgmt_vals) > 0:
                mean_mgmt = mgmt_vals.mean()
                std_mgmt = mgmt_vals.std()
                print(f"  {model:8s} | Mean mgmt accuracy: {mean_mgmt:.0%} (SD={std_mgmt:.0%}, n={len(mgmt_vals)})")
    else:
        print("  (consensus_mgmt_acc column not found)")

    # ── McNemar's test for framing effect ──
    print("\n── McNemar's Test: Physician vs Patient Frame ──")
    paired = df[df["category"] != "E"].copy()
    for model in paired["model"].unique():
        phys = paired[(paired["model"] == model) & (paired["frame"] == "physician")].sort_values("scenario_id")
        pat = paired[(paired["model"] == model) & (paired["frame"] == "patient")].sort_values("scenario_id")
        
        if len(phys) != len(pat):
            print(f"  {model}: Unequal paired observations — skipping")
            continue
        
        phys_correct = (phys["triage_score"].astype(float).values == 2)
        pat_correct = (pat["triage_score"].astype(float).values == 2)
        
        # McNemar contingency
        b = ((phys_correct) & (~pat_correct)).sum()  # Physician correct, patient wrong
        c = ((~phys_correct) & (pat_correct)).sum()   # Physician wrong, patient correct
        
        if b + c > 0:
            # McNemar's test (exact binomial for small samples)
            p_value = stats.binomtest(b, b + c, 0.5).pvalue
            print(f"  {model:8s} | b={b} (phys✓ pat✗), c={c} (phys✗ pat✓) | p={p_value:.4f}")
        else:
            print(f"  {model:8s} | No discordant pairs")

    # ── Framing gap ──
    print("\n── Framing Gap (% of scenarios where patient frame under-triaged vs physician) ──")
    for model in paired["model"].unique():
        phys = paired[(paired["model"] == model) & (paired["frame"] == "physician")].sort_values("scenario_id")
        pat = paired[(paired["model"] == model) & (paired["frame"] == "patient")].sort_values("scenario_id")
        
        if len(phys) != len(pat):
            continue
        
        # Compare triage levels
        discordant = 0
        patient_under = 0
        for (_, p), (_, q) in zip(phys.iterrows(), pat.iterrows()):
            p_level = TRIAGE_MAP.get(p.get("consensus_triage_level", ""), -1)
            q_level = TRIAGE_MAP.get(q.get("consensus_triage_level", ""), -1)
            if p_level != q_level:
                discordant += 1
                if q_level < p_level:
                    patient_under += 1
        
        n = len(phys)
        print(f"  {model:8s} | Discordant: {discordant}/{n} ({discordant/n*100:.0f}%) | "
              f"Patient under-triaged: {patient_under}/{n} ({patient_under/n*100:.0f}%)")

    # ── Expected Calibration Error (physician frame only) ──
    print("\n── Expected Calibration Error (ECE) — Physician Frame Only ──")
    print("  Note: Confidence elicited only in physician frame (patients don't ask 'how confident are you').")
    phys_for_ece = df[df["frame"] == "physician"]
    for model in phys_for_ece["model"].unique():
        sub = phys_for_ece[phys_for_ece["model"] == model].copy()
        if sub.empty or "confidence" not in sub.columns:
            continue
        sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
        sub = sub.dropna(subset=["confidence"])
        if sub.empty:
            continue
        sub["conf_norm"] = sub["confidence"] / 10
        sub["correct"] = sub["triage_score"].astype(float) == 2
        
        bins = [0, 0.4, 0.7, 1.01]
        sub["bin"] = pd.cut(sub["conf_norm"], bins=bins, include_lowest=True)
        grouped = sub.groupby("bin", observed=True).agg(
            mean_conf=("conf_norm", "mean"),
            mean_acc=("correct", "mean"),
            count=("correct", "count"),
        ).dropna()
        
        if len(grouped) > 0:
            n_total = grouped["count"].sum()
            ece = (grouped["count"] / n_total * (grouped["mean_conf"] - grouped["mean_acc"]).abs()).sum()
            print(f"  {model:8s} | ECE = {ece:.3f} (n={int(n_total)})")

    # ── Inter-rater reliability placeholder ──
    print("\n── Inter-Rater Reliability ──")
    print("  (Compute Fleiss' kappa from the 3 individual scorer sheets)")
    print("  Use: from statsmodels.stats.inter_rater import fleiss_kappa")
    print("  Or: pip install statsmodels && python compute_irr.py")

    # ── SENSITIVITY ANALYSES ──
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSES")
    print("="*60)

    # ── S1: Caller specialty comparison (Category B ED docs vs Category C GPs) ──
    print("\n── S1: Caller Specialty Effect (ED Physician vs GP Dentist, physician frame only) ──")
    phys_only = df[df["frame"] == "physician"]
    for model in phys_only["model"].unique():
        cat_b = phys_only[(phys_only["model"] == model) & (phys_only["category"] == "B")]
        cat_c = phys_only[(phys_only["model"] == model) & (phys_only["category"] == "C")]
        if cat_b.empty or cat_c.empty:
            continue
        b_acc = (cat_b["triage_score"].astype(float) == 2).mean() * 100
        c_acc = (cat_c["triage_score"].astype(float) == 2).mean() * 100
        b_mgmt = cat_b["consensus_mgmt_acc"].astype(float).mean() * 100 if "consensus_mgmt_acc" in cat_b.columns else float('nan')
        c_mgmt = cat_c["consensus_mgmt_acc"].astype(float).mean() * 100 if "consensus_mgmt_acc" in cat_c.columns else float('nan')
        print(f"  {model:8s} | Cat B (ED physician): triage acc={b_acc:.0f}%, mgmt acc={b_mgmt:.0f}%")
        print(f"  {model:8s} | Cat C (GP dentist):   triage acc={c_acc:.0f}%, mgmt acc={c_mgmt:.0f}%")
        print(f"  {model:8s} | Delta: {b_acc - c_acc:+.0f} percentage points triage accuracy")
    print("  Note: n=5 per category — descriptive only, not inferential.")

    # ── S2: Excluding Scenario 9 (redirect/wrong consult) ──
    print("\n── S2: Primary Analysis Excluding Scenario 9 (Redirect) ──")
    df_no9 = df[df["scenario_id"] != 9].copy()
    paired_no9 = df_no9[df_no9["category"] != "E"]
    for model in paired_no9["model"].unique():
        phys = paired_no9[(paired_no9["model"] == model) & (paired_no9["frame"] == "physician")].sort_values("scenario_id")
        pat = paired_no9[(paired_no9["model"] == model) & (paired_no9["frame"] == "patient")].sort_values("scenario_id")
        if len(phys) != len(pat) or len(phys) == 0:
            continue
        phys_correct = (phys["triage_score"].astype(float).values == 2)
        pat_correct = (pat["triage_score"].astype(float).values == 2)
        b = ((phys_correct) & (~pat_correct)).sum()
        c = ((~phys_correct) & (pat_correct)).sum()
        phys_acc = phys_correct.mean() * 100
        pat_acc = pat_correct.mean() * 100
        p_str = ""
        if b + c > 0:
            p_value = stats.binomtest(b, b + c, 0.5).pvalue
            p_str = f" | McNemar p={p_value:.4f}"
        print(f"  {model:8s} | Phys acc={phys_acc:.0f}%, Pat acc={pat_acc:.0f}%{p_str} (n={len(phys)} pairs, excl. S9)")

    # ── S3: Category E standalone analysis ──
    print("\n── S3: Category E (Anesthesia Emergencies) — Standalone Analysis ──")
    cat_e = df[df["category"] == "E"]
    if not cat_e.empty:
        for model in cat_e["model"].unique():
            sub = cat_e[cat_e["model"] == model]
            n = len(sub)
            correct = (sub["triage_score"].astype(float) == 2).sum()
            mgmt = sub["consensus_mgmt_acc"].astype(float).mean() if "consensus_mgmt_acc" in sub.columns else float('nan')
            dangerous = (sub["dangerous_rec_(consensus)"] == "Y").sum() if "dangerous_rec_(consensus)" in sub.columns else 0
            print(f"  {model:8s} | Triage correct: {correct}/{n} | "
                  f"Mean mgmt accuracy: {mgmt:.0%} | "
                  f"Dangerous recs: {dangerous}/{n}")
        
        # Per-scenario breakdown for Category E
        print("\n  Per-scenario breakdown (Category E):")
        for _, row in cat_e.iterrows():
            score = row.get("triage_score", "?")
            mgmt = row.get("consensus_mgmt_acc", "?")
            model = row.get("model", "?")
            name = row.get("scenario_name", f"S{row.get('scenario_id', '?')}")
            dangerous = row.get("dangerous_rec_(consensus)", "N")
            errors = row.get("error_types_(consensus)", "")
            flag = " ⚠️DANGEROUS" if dangerous == "Y" else ""
            print(f"    S{row.get('scenario_id', '?'):2d} ({name[:30]:30s}) | {model:8s} | "
                  f"triage={score} | mgmt={mgmt}{flag}"
                  + (f" | errors: {errors}" if errors and str(errors) != "nan" else ""))
    else:
        print("  No Category E data found.")

    # ── S4: Information-seeking behavior in patient frame ──
    print("\n── S4: Information-Seeking Behavior (Patient Frame Only) ──")
    patient_data = df[df["frame"] == "patient"]
    if "asked_clarifying_questions" in patient_data.columns or "information_seeking" in patient_data.columns:
        # Try common column name variations
        q_col = None
        for candidate in ["asked_clarifying_questions", "information_seeking", "asks_questions"]:
            if candidate in patient_data.columns:
                q_col = candidate
                break
        if q_col:
            for model in patient_data["model"].unique():
                sub = patient_data[patient_data["model"] == model]
                asked = (sub[q_col].astype(str).str.upper() == "Y").sum()
                total = len(sub)
                # Compare accuracy for asked vs didn't ask
                asked_rows = sub[sub[q_col].astype(str).str.upper() == "Y"]
                not_asked = sub[sub[q_col].astype(str).str.upper() != "Y"]
                asked_acc = (asked_rows["triage_score"].astype(float) == 2).mean() * 100 if len(asked_rows) > 0 else 0
                not_acc = (not_asked["triage_score"].astype(float) == 2).mean() * 100 if len(not_asked) > 0 else 0
                print(f"  {model:8s} | Asked questions: {asked}/{total} ({asked/total*100:.0f}%) | "
                      f"Accuracy when asked: {asked_acc:.0f}% vs didn't ask: {not_acc:.0f}%")
        else:
            print("  (Column not found — score this manually in the spreadsheet)")
    else:
        print("  (Information-seeking data not found in CSV — score this in the spreadsheet)")

    # ── S5: Confidence-accuracy correlation (physician frame only) ──
    print("\n── S5: Confidence-Accuracy Correlation (Spearman's) — Physician Frame Only ──")
    phys_for_corr = df[df["frame"] == "physician"]
    if "confidence" in phys_for_corr.columns:
        for model in phys_for_corr["model"].unique():
            sub = phys_for_corr[phys_for_corr["model"] == model].copy()
            sub["conf"] = pd.to_numeric(sub["confidence"], errors="coerce")
            sub["correct"] = (sub["triage_score"].astype(float) == 2).astype(int)
            sub = sub.dropna(subset=["conf"])
            if len(sub) > 5:
                rho, p = stats.spearmanr(sub["conf"], sub["correct"])
                print(f"  {model:8s} | Spearman rho={rho:.3f}, p={p:.4f} (n={len(sub)})")

    print("\n" + "="*60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <consensus_scores.csv>")
        print("\nTo generate a template CSV from the scoring spreadsheet:")
        print("  1. Complete scoring in the XLSX")
        print("  2. Export the 'Consensus Scores' tab as CSV")
        print("  3. Run: python analyze.py data/consensus_scores.csv")
        
        # Generate a sample CSV template
        sample_path = Path("data/consensus_scores_TEMPLATE.csv")
        cols = [
            "run_id", "scenario_id", "category", "scenario_name", "model", "frame",
            "correct_triage", "consensus_triage_level", "triage_score",
            "under_triage_(consensus)", "over_triage_(consensus)",
            "consensus_mgmt_acc", "dangerous_rec_(consensus)",
            "safe_uncertainty_(consensus)", "error_types_(consensus)", "confidence"
        ]
        pd.DataFrame(columns=cols).to_csv(sample_path, index=False)
        print(f"\n  Template CSV saved to {sample_path}")
        return

    filepath = sys.argv[1]
    print(f"Loading scored data from {filepath}...")
    df = load_data(filepath)
    print(f"Loaded {len(df)} scored rows")

    # Load ground truth from scenario JSON files
    print(f"Loading ground truth from prompts/*.json...")
    scenarios = load_scenarios()
    if scenarios:
        gt = {s["id"]: s for s in scenarios}
        print(f"Loaded {len(gt)} scenarios with scoring keys")
        
        # Merge critical element counts and ground truth into df for reference
        if "scenario_id" in df.columns:
            df["num_critical_elements"] = df["scenario_id"].map(
                lambda sid: len(gt[sid]["critical_elements"]) if sid in gt else None
            )
    else:
        print("WARNING: No scenario JSONs found in prompts/ — ground truth not loaded")

    print(f"Models: {df['model'].unique()}")
    print(f"Frames: {df['frame'].unique()}")
    print(f"Categories: {df['category'].unique()}")

    # Generate all figures
    print("\nGenerating figures...")
    fig1_triage_confusion(df)
    fig2_framing_gap(df)
    fig3_calibration(df)
    fig4_undertriage_by_category(df)
    fig5_safe_uncertainty(df)
    fig6_error_taxonomy(df)
    
    # Run statistical analyses
    compute_statistics(df)
    
    # Print ground truth summary for reference
    if scenarios:
        print("\n── Ground Truth Summary (from prompts/*.json) ──")
        for s in scenarios:
            print(f"  S{s['id']:02d} | {s['category']} | {s['correct_triage']:15s} | "
                  f"{len(s['critical_elements'])} elements | {s['name'][:40]}")
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
