"""
AAOMS 2026 AI Triage Study — Inter-Rater Reliability
Computes Fleiss' kappa across 3 scorers for triage level, under-triage, and safe uncertainty.

Usage:
    python compute_irr.py data/scorer1.csv data/scorer2.csv data/scorer3.csv
    
    Each CSV should have columns: run_id, triage_score, under_triage, safe_uncertainty
"""

import sys
import numpy as np
import pandas as pd


def fleiss_kappa(M):
    """
    Compute Fleiss' kappa for inter-rater reliability.
    M: numpy array, shape (n_subjects, n_categories)
        M[i,j] = number of raters who assigned subject i to category j
    """
    N, k = M.shape
    n = M.sum(axis=1)[0]  # number of raters (should be constant)
    
    # Proportion of assignments to each category
    p = M.sum(axis=0) / (N * n)
    
    # Per-subject agreement
    P = (M * M).sum(axis=1) - n
    P = P / (n * (n - 1))
    
    P_bar = P.mean()
    P_e = (p * p).sum()
    
    if P_e == 1:
        return 1.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_irr(scorer_files):
    """Load scorer data and compute Fleiss' kappa."""
    dfs = []
    for f in scorer_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("\n", "_")
        dfs.append(df)

    n_scorers = len(dfs)
    n_runs = len(dfs[0])
    
    print(f"Scorers: {n_scorers}")
    print(f"Runs per scorer: {n_runs}")

    # ── Triage score (0, 1, 2) ──
    print("\n── Fleiss' Kappa: Triage Score ──")
    categories = [0, 1, 2]
    M = np.zeros((n_runs, len(categories)), dtype=int)
    for df in dfs:
        for i, row in df.iterrows():
            score = int(row.get("triage_score", -1))
            if score in categories:
                M[i, categories.index(score)] += 1
    
    kappa = fleiss_kappa(M)
    print(f"  Fleiss' kappa = {kappa:.3f}")
    if kappa > 0.8:
        print(f"  Interpretation: Excellent agreement")
    elif kappa > 0.6:
        print(f"  Interpretation: Substantial agreement")
    elif kappa > 0.4:
        print(f"  Interpretation: Moderate agreement")
    else:
        print(f"  Interpretation: Fair or poor agreement")

    # ── Under-triage (Y/N) ──
    print("\n── Fleiss' Kappa: Under-Triage ──")
    categories_yn = ["Y", "N"]
    M_ut = np.zeros((n_runs, 2), dtype=int)
    for df in dfs:
        for i, row in df.iterrows():
            val = str(row.get("under_triage", "")).strip().upper()
            if val in categories_yn:
                M_ut[i, categories_yn.index(val)] += 1
    
    if M_ut.sum() > 0:
        kappa_ut = fleiss_kappa(M_ut)
        print(f"  Fleiss' kappa = {kappa_ut:.3f}")
    else:
        print(f"  No data")

    # ── Pairwise Cohen's kappa ──
    print("\n── Pairwise Cohen's Kappa (Triage Score) ──")
    from itertools import combinations
    for (i, j) in combinations(range(n_scorers), 2):
        s1 = dfs[i]["triage_score"].astype(int).values
        s2 = dfs[j]["triage_score"].astype(int).values
        
        # Cohen's kappa
        n = len(s1)
        observed_agree = (s1 == s2).sum() / n
        
        cats = sorted(set(s1) | set(s2))
        expected_agree = sum(
            ((s1 == c).sum() / n) * ((s2 == c).sum() / n) for c in cats
        )
        
        if expected_agree == 1:
            ck = 1.0
        else:
            ck = (observed_agree - expected_agree) / (1 - expected_agree)
        
        print(f"  Scorer {i+1} vs Scorer {j+1}: κ = {ck:.3f} (agreement = {observed_agree*100:.0f}%)")

    # ── Agreement rate ──
    print("\n── Simple Agreement Rate ──")
    agree_count = 0
    for idx in range(n_runs):
        scores = [int(df.iloc[idx].get("triage_score", -1)) for df in dfs]
        if len(set(scores)) == 1:
            agree_count += 1
    print(f"  All 3 agree: {agree_count}/{n_runs} ({agree_count/n_runs*100:.0f}%)")
    
    majority = 0
    for idx in range(n_runs):
        scores = [int(df.iloc[idx].get("triage_score", -1)) for df in dfs]
        from collections import Counter
        c = Counter(scores)
        if c.most_common(1)[0][1] >= 2:
            majority += 1
    print(f"  Majority (2/3) agree: {majority}/{n_runs} ({majority/n_runs*100:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compute_irr.py scorer1.csv scorer2.csv scorer3.csv")
        sys.exit(1)
    compute_irr(sys.argv[1:4])
