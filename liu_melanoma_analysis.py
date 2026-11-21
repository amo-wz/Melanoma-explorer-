#!/usr/bin/env python3
"""
Liu et al. 2019 (Nat Med) anti-PD-1 melanoma cohort: survival and UV-signature analysis.

Data: cBioPortal study "mel_dfci_2019" -> Summary tab -> download clinical data (.tsv).
Usage:
    python3 liu_melanoma_analysis.py mel_dfci_2019_clinical_data.tsv [--out results]

Outputs (in --out):
    results.json           all summary statistics
    survival_table.csv     Kaplan-Meier medians (95% CI), 12- and 24-month OS by subgroup
    uv_by_primary.csv      UV-induced mutation share and TMB by primary type
    fig_uv.png             UV share by primary type (strip plot, median bars)
    fig_km.png             Kaplan-Meier OS by brain-metastasis status
    cohort.json            de-identified per-patient table used by the interactive page

Methods:
    - Overall survival from start of anti-PD-1; event = "1:DECEASED".
    - Kaplan-Meier estimator; 95% CI by Greenwood variance on the log(-log) scale.
    - Median CI = times where the lower/upper confidence band first reaches 0.5.
    - Two-group comparison by log-rank test (1 df).
    - UV share = UV-induced mutations / total mutations (per tumour).
    - Group comparisons of UV share: Kruskal-Wallis (all groups), Mann-Whitney U (pairs).
Requires: pandas, numpy, scipy, matplotlib.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import chi2, kruskal, mannwhitneyu


# ---------------------------------------------------------------- statistics
def kaplan_meier(time, event):
    """Return a DataFrame of event times with S(t) and log(-log) 95% CI."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    rows, S, var_sum = [(0.0, 1.0, 1.0, 1.0)], 1.0, 0.0
    for ti in np.unique(t[e == 1]):
        n_risk = (t >= ti).sum()
        d = ((t == ti) & (e == 1)).sum()
        S *= 1 - d / n_risk
        if n_risk - d > 0:
            var_sum += d / (n_risk * (n_risk - d))
        if 0 < S < 1:
            se = np.sqrt(var_sum) / abs(np.log(S))
            lo, hi = S ** np.exp(1.96 * se), S ** np.exp(-1.96 * se)
        else:
            lo = hi = S
        rows.append((ti, S, lo, hi))
    return pd.DataFrame(rows, columns=["t", "S", "lo", "hi"])


def km_median(curve):
    """Median survival and 95% CI (None = not reached)."""
    def first_below(col):
        hit = curve[curve[col] <= 0.5]
        return None if hit.empty else float(hit.t.iloc[0])
    return first_below("S"), first_below("lo"), first_below("hi")


def surv_at(curve, month):
    return float(curve[curve.t <= month].S.iloc[-1])


def logrank_p(t1, e1, t2, e2):
    t = np.r_[t1, t2]; e = np.r_[e1, e2]
    g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    O = E = V = 0.0
    for ti in np.unique(t[e == 1]):
        at = t >= ti
        n, n1 = at.sum(), (at & (g == 0)).sum()
        d = ((t == ti) & (e == 1)).sum()
        d1 = ((t == ti) & (e == 1) & (g == 0)).sum()
        O += d1
        E += d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return float(1 - chi2.cdf((O - E) ** 2 / V, 1))


# ---------------------------------------------------------------- data
def load_liu(path):
    d = pd.read_csv(path, sep="\t")
    d["os"] = d["Overall Survival (Months)"].astype(float)
    d["event"] = d["Overall Survival Status"].str.startswith("1").astype(int)
    d["uv_share"] = (d["UV-Induced Mutations"].astype(float)
                     / d["Total Mutations"].astype(float)).clip(0, 1)
    d["tmb"] = d["TMB (nonsynonymous)"].astype(float)
    return d


def subgroup_row(name, sub):
    c = kaplan_meier(sub.os, sub.event)
    med, lo, hi = km_median(c)
    return dict(subgroup=name, n=len(sub), deaths=int(sub.event.sum()),
                median_months=med, ci_low=lo, ci_high=hi,
                os_12m=round(surv_at(c, 12), 3), os_24m=round(surv_at(c, 24), 3))


def analyse(d):
    res = {"n": len(d), "primary_counts": d["Primary Type"].value_counts().to_dict()}

    # UV signature by primary type
    uv_rows = []
    for k, g in d.groupby("Primary Type"):
        uv_rows.append(dict(primary_type=k, n=len(g),
                            uv_median=round(g.uv_share.median(), 3),
                            uv_q1=round(g.uv_share.quantile(.25), 3),
                            uv_q3=round(g.uv_share.quantile(.75), 3),
                            uv_high_pct=round(100 * (g.uv_share >= 0.6).mean(), 1),
                            tmb_median=round(g.tmb.median(), 1)))
    res["uv_by_primary"] = uv_rows
    groups = [g.uv_share.dropna() for _, g in d.groupby("Primary Type")]
    occ = d[d["Primary Type"] == "occult"].uv_share
    res["uv_kruskal_p"] = float(kruskal(*groups).pvalue)
    res["uv_occult_vs_skin_p"] = float(mannwhitneyu(occ, d[d["Primary Type"] == "skin"].uv_share).pvalue)
    res["uv_occult_vs_acral_mucosal_p"] = float(mannwhitneyu(
        occ, d[d["Primary Type"].isin(["acral", "mucosal"])].uv_share).pvalue)

    # Survival
    yes = lambda col: d[col] == "Yes"
    subs = [
        ("All patients", d),
        ("No brain metastases", d[~yes("Brain Metastasis")]),
        ("Brain metastases", d[yes("Brain Metastasis")]),
        ("Brain + visceral + bone metastases",
         d[yes("Brain Metastasis") & yes("Liver Visceral Metastasis") & yes("Bone Metastasis")]),
        ("Elevated LDH", d[yes("LDH Elevated")]),
        ("M1c", d[d["M Stage"] == "M1c"]),
        ("Cutaneous primary", d[d["Primary Type"] == "skin"]),
        ("Occult primary", d[d["Primary Type"] == "occult"]),
    ]
    res["survival"] = [subgroup_row(n, s) for n, s in subs]
    bm, nbm = d[yes("Brain Metastasis")], d[~yes("Brain Metastasis")]
    res["logrank_brain_p"] = logrank_p(bm.os, bm.event, nbm.os, nbm.event)
    o, s = d[d["Primary Type"] == "occult"], d[d["Primary Type"] == "skin"]
    res["logrank_occult_vs_skin_p"] = logrank_p(o.os, o.event, s.os, s.event)
    return res


# ---------------------------------------------------------------- figures
def make_figures(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    INK, SEC, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

    # UV strip plot
    order = [("skin", "Cutaneous"), ("occult", "Occult primary"), ("acral", "Acral"), ("mucosal", "Mucosal")]
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=300)
    rng = np.random.default_rng(7)
    for i, (k, lab) in enumerate(order):
        v = d.loc[d["Primary Type"] == k, "uv_share"].dropna().values
        ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=22, alpha=.75, edgecolor="white", lw=.6,
                   color=BLUE if k == "occult" else "#9a9892", zorder=3)
        m = np.median(v)
        ax.plot([i - .3, i + .3], [m, m], color=INK, lw=2.2, zorder=4)
        ax.text(i + .33, m, f"{m:.2f}", va="center", fontsize=11, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{lab}\n(n = {(d['Primary Type'] == k).sum()})" for k, lab in order])
    ax.set_ylim(0, 1.02); ax.set_ylabel("Share of mutations that are UV-induced")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title("Occult-primary melanomas carry a cutaneous-type UV signature",
                 loc="left", fontsize=14, color=INK, pad=22)
    ax.text(0, 1.035, "Liu et al., Nat Med 2019 (cBioPortal mel_dfci_2019). Bar = median.",
            transform=ax.transAxes, fontsize=10, color=SEC)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_uv.png")); plt.close(fig)

    # Kaplan-Meier
    def draw(ax, sub, col, label):
        c = kaplan_meier(sub.os, sub.event)
        ax.step(c.t, c.S, where="post", color=col, lw=2, label=label)
        ax.plot([c.t.iloc[-1], sub.os.max()], [c.S.iloc[-1]] * 2, color=col, lw=2)
        cens = sub[sub.event == 0].os.values
        ax.scatter(cens, [c[c.t <= x].S.iloc[-1] for x in cens], marker="|", s=60, color=col, zorder=3)

    yes = lambda col: d[col] == "Yes"
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=300)
    groups = [(d[~yes("Brain Metastasis")], BLUE, "No brain metastases"),
              (d[yes("Brain Metastasis")], ORANGE, "Brain metastases"),
              (d[yes("Brain Metastasis") & yes("Liver Visceral Metastasis") & yes("Bone Metastasis")],
               AQUA, "Brain + visceral + bone, subset")]
    for sub, col, lab in groups:
        med = km_median(kaplan_meier(sub.os, sub.event))[0]
        draw(ax, sub, col, f"{lab} (n = {len(sub)}): median {med:.1f} mo")
    ax.axhline(.5, color=GRID, ls="--", zorder=0)
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months from start of anti-PD-1"); ax.set_ylabel("Overall survival")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    ax.set_title("Brain metastases shorten survival on anti-PD-1", loc="left", fontsize=14, color=INK, pad=22)
    ax.text(0, 1.035, "Kaplan-Meier, Liu et al. 2019 (n = 144). Ticks = censored.",
            transform=ax.transAxes, fontsize=10, color=SEC)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_km.png")); plt.close(fig)


# ---------------------------------------------------------------- export for web page
def export_cohort(d, out):
    keep = {"Primary Type": "primary", "M Stage": "m", "Brain Metastasis": "brain",
            "Liver Visceral Metastasis": "visceral", "Bone Metastasis": "bone",
            "Lung Metastasis": "lung", "LN Metastasis": "node", "LDH Elevated": "ldh",
            "Best Radiographic Response (RECIST 1.1)": "response"}
    rows = []
    for _, r in d.iterrows():
        row = {v: r[k] for k, v in keep.items()}
        row.update(os=round(r.os, 2), ev=int(r.event),
                   uv=None if pd.isna(r.uv_share) else round(float(r.uv_share), 3),
                   tmb=None if pd.isna(r.tmb) else round(float(r.tmb), 2))
        rows.append(row)
    json.dump(rows, open(os.path.join(out, "cohort.json"), "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", help="cBioPortal clinical data file for mel_dfci_2019")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_liu(a.tsv)
    res = analyse(d)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    pd.DataFrame(res["survival"]).to_csv(os.path.join(a.out, "survival_table.csv"), index=False)
    pd.DataFrame(res["uv_by_primary"]).to_csv(os.path.join(a.out, "uv_by_primary.csv"), index=False)
    make_figures(d, a.out)
    export_cohort(d, a.out)
    print(pd.DataFrame(res["survival"]).to_string(index=False))
    print(f"\nLog-rank brain vs no brain: p = {res['logrank_brain_p']:.3f}")
    print(pd.DataFrame(res["uv_by_primary"]).to_string(index=False))
    print(f"UV occult vs cutaneous p = {res['uv_occult_vs_skin_p']:.2f}; "
          f"occult vs acral/mucosal p = {res['uv_occult_vs_acral_mucosal_p']:.4f}")


if __name__ == "__main__":
    main()
