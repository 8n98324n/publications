# -*- coding: utf-8 -*-
"""fig_part2.py: Publication-grade 300-DPI matplotlib generators for Figures 9-13, C1-C2, and D1-D4."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KG = "#2CA25F"
C_BS = "#2B7BBA"


def plot_fig09(res_s07: dict, out_path):
    """Figure 9: Internal consistency (Cronbach's alpha and McDonald's omega) with bootstrap 95% CIs."""
    st = res_s07["summary_table"].set_index("condition")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    for y_base, cond, col, lbl in [(1.0, "KG-LLM", C_KG, "KG-LLM"), (0.0, "base_LLM", C_BS, "base LLM")]:
        r = st.loc[cond]
        ax.errorbar(r["alpha_40"], y_base + 0.14, xerr=[[r["alpha_40"] - r["alpha_ci_low"]], [r["alpha_ci_high"] - r["alpha_40"]]],
                    fmt="o", color=col, capsize=4, markersize=8.5, lw=1.8)
        ax.text(r["alpha_40"], y_base + 0.28, f"α = {r['alpha_40']:.2f} [{r['alpha_ci_low']:.2f}, {r['alpha_ci_high']:.2f}]", ha="center", color=col, fontsize=9.6)

        ax.errorbar(r["omega_40"], y_base - 0.14, xerr=[[r["omega_40"] - r["omega_ci_low"]], [r["omega_ci_high"] - r["omega_40"]]],
                    fmt="s", mfc="white", mec=col, ecolor=col, capsize=4, markersize=8.5, mew=1.8, lw=1.8)
        ax.text(r["omega_40"], y_base - 0.36, f"ω = {r['omega_40']:.2f} [{r['omega_ci_low']:.2f}, {r['omega_ci_high']:.2f}]", ha="center", color=col, fontsize=9.6)

    ax.axvline(0.70, ls="--", color="#777777", lw=1.2)
    ax.text(0.692, -0.45, ".70 acceptable", ha="right", color="#666666", fontsize=9.3)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["base LLM", "KG-LLM"])
    ax.set_xlim(0.25, 0.95)
    ax.set_ylim(-0.55, 1.48)
    ax.set_xlabel("Reliability coefficient")
    ax.set_title("α (circles) and ω (squares), 95% bootstrap CI", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig10(res_s07: dict, res_s09: dict, out_path):
    """Figure 10: Polarity reliability and pilot-to-formal ML scoring-key TVD."""
    st = res_s07["summary_table"].set_index("condition")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 4.2), gridspec_kw={"width_ratios": [1.15, 0.95], "wspace": 0.35})

    x = np.arange(4)
    w = 0.36
    kg_v = [st.loc["KG-LLM", "alpha_40"], st.loc["KG-LLM", "alpha_ml20"], st.loc["KG-LLM", "alpha_ll20"], st.loc["KG-LLM", "sb_proj_20"]]
    bs_v = [st.loc["base_LLM", "alpha_40"], st.loc["base_LLM", "alpha_ml20"], st.loc["base_LLM", "alpha_ll20"], st.loc["base_LLM", "sb_proj_20"]]
    ax_a.bar(x - w / 2, kg_v, width=w, color=C_KG, label="KG-LLM")
    ax_a.bar(x + w / 2, bs_v, width=w, color=C_BS, label="base LLM")
    ax_a.axhline(0.70, ls="--", color="#777777", lw=1.1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["All", "ML", "LL", "SB"])
    ax_a.set_ylim(0.0, 1.10)
    ax_a.set_ylabel("Cronbach's α")
    ax_a.set_title("(a) α by polarity", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="upper left")

    rng = np.random.default_rng(20240501)
    tk, tb = res_s09["tvd_ml_kg"], res_s09["tvd_ml_base"]
    ax_b.scatter(np.zeros(len(tk)) + rng.uniform(-0.10, 0.10, len(tk)), tk, color=C_KG, alpha=0.65, s=34)
    ax_b.scatter(np.ones(len(tb)) + rng.uniform(-0.10, 0.10, len(tb)), tb, color=C_BS, alpha=0.65, s=34)
    ax_b.errorbar([0.26], [tk.mean()], yerr=[tk.std(ddof=1)], fmt="o", color="#111111", capsize=4, lw=1.8)
    ax_b.errorbar([1.26], [tb.mean()], yerr=[tb.std(ddof=1)], fmt="o", color="#111111", capsize=4, lw=1.8)
    ax_b.text(-0.22, 0.43, f"g = {res_s09['wt_tvd_ml']['hedges_g']:.2f}", fontsize=9.8)
    ax_b.set_xticks([0.1, 1.1])
    ax_b.set_xticklabels(["KG-LLM", "base LLM"])
    ax_b.set_xlim(-0.35, 1.55)
    ax_b.set_ylim(0.11, 0.46)
    ax_b.set_ylabel("Pilot-to-formal TVD")
    ax_b.set_title("(b) Key drift", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig11(res_s08: dict, res_s09: dict, out_path):
    """Figure 11: Facet-level convergent validity and behavioral coping-strategy choices."""
    fac = res_s08["facets_polarity"].set_index("criterion").loc[
        ["5x5RS_Total", "Adaptability", "Emotion_Regulation", "Optimism", "Self_Efficacy", "Social_Support", "PA"]
    ]
    ch = res_s09["choice_table"].set_index("strategy")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.8, 4.8), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.55})

    lbls_a = ["5×5RS total", "Adaptability", "Emotion regulation", "Optimism", "Self-efficacy", "Social support", "PA"]
    ya = np.arange(len(lbls_a))[::-1]
    for i, (_, r) in enumerate(fac.iterrows()):
        yi = ya[i]
        ax_a.plot([r["r40_llm"], r["r40_kg"]], [yi, yi], color="#AAAAAA", lw=1.8, zorder=1)
        ax_a.scatter([r["r40_llm"]], [yi], color=C_BS, s=55, zorder=3)
        ax_a.scatter([r["r40_kg"]], [yi], color=C_KG, s=55, zorder=3)
        if r["p40"] < 0.05:
            ax_a.text(max(r["r40_kg"], r["r40_llm"]) + 0.012, yi, f"Z = {r['z40']:.2f}*", va="center", fontsize=9.2)
    ax_a.scatter([], [], color=C_KG, s=50, label="KG-LLM")
    ax_a.scatter([], [], color=C_BS, s=50, label="base LLM")
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(lbls_a)
    ax_a.set_xlim(0.20, 0.58)
    ax_a.set_xlabel("r with the 40-indicator SJT score")
    ax_a.set_title("(a) Facet-level\nconvergent validity", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    strats = ["Adaptability", "Emotion_Regulation", "Optimism", "Self_Efficacy", "Social_Support"]
    pretty_s = ["Adaptability", "Emotion regulation", "Optimism", "Self-efficacy", "Social support"]
    yb = np.arange(5)[::-1]
    h = 0.35
    ml_v = [ch.loc[s, "ml_mean_pct"] for s in strats]
    ll_v = [ch.loc[s, "ll_mean_pct"] for s in strats]
    ax_b.barh(yb + h / 2, ml_v, height=h, color="#636363", label="Most likely")
    ax_b.barh(yb - h / 2, ll_v, height=h, color="#BDBDBD", label="Least likely")
    for i in range(5):
        ax_b.text(ml_v[i] + 1.0, yb[i] + h / 2, f"{ml_v[i]:.1f}", va="center", fontsize=9.0)
        ax_b.text(ll_v[i] + 1.0, yb[i] - h / 2, f"{ll_v[i]:.1f}", va="center", fontsize=9.0)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(pretty_s)
    ax_b.set_xlim(0, 72)
    ax_b.set_xlabel("Share of choices (%), both SJTs")
    ax_b.set_title("(b) Coping-strategy\nchoices", loc="left", fontsize=11, fontweight="bold")
    ax_b.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig12(res_s08: dict, out_path):
    """Figure 12: Criterion correlations of both SJTs with Steiger's Z contrasts."""
    d4a = res_s08["table_d4a"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(d4a))[::-1]

    for i, (_, r) in enumerate(d4a.iterrows()):
        yi = y[i]
        ax.plot([r["r_llm"], r["r_kg"]], [yi, yi], color="#CCCCCC", lw=2.4, zorder=1)
        ax.scatter([r["r_llm"]], [yi], color=C_BS, s=85, zorder=3)
        ax.scatter([r["r_kg"]], [yi], color=C_KG, s=85, zorder=3)
        sig = r["p_diff"] < 0.05
        star = "*" if sig else ""
        txt = f"Z = {r['steiger_z']:+.2f}{star}  (p = {r['p_diff']:.3f})"
        x_txt = max(r["r_kg"], r["r_llm"]) + 0.03
        ax.text(x_txt, yi, txt, va="center", fontsize=9.5, fontweight="bold" if sig else "normal", color="#111111" if sig else "#666666")

    ax.scatter([], [], color=C_KG, s=75, label="KG-LLM")
    ax.scatter([], [], color=C_BS, s=75, label="base LLM")
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(["5×5RS", "PA", "NA", "Depression", "Anxiety", "Stress"])
    ax.set_xlim(-0.55, 0.76)
    ax.set_xlabel("Correlation with criterion (r), N = 150")
    ax.set_title("Steiger's Z, * p < .05", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig13(res_s08: dict, out_path):
    """Figure 13: Exploratory moderation (LL, Domain x Polarity) and cross-validated hybrid forms."""
    fac = res_s08["facets_polarity"].set_index("criterion")
    mod8 = res_s08["moderation_8tests"].set_index("subset")
    d4c = res_s08["table_d4c"].set_index("architecture")

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.2), gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) 20 LL indicators
    ll_keys = ["5x5RS_Total", "Self_Efficacy", "Social_Support", "PA", "Depression", "Stress"]
    ll_lbls = ["5×5RS", "SE", "SS", "PA", "Dep.", "Str."]
    x6 = np.arange(6)
    w = 0.36
    sub_ll = fac.loc[ll_keys]
    ax_a.bar(x6 - w / 2, sub_ll["rll_kg"], width=w, color=C_KG, label="KG-LLM")
    ax_a.bar(x6 + w / 2, sub_ll["rll_llm"], width=w, color=C_BS, label="base LLM")
    for i, (_, r) in enumerate(sub_ll.iterrows()):
        if r["pll"] < 0.01:
            ax_a.text(x6[i], max(r["rll_kg"], r["rll_llm"], 0) + 0.02, "**", ha="center", fontsize=10)
        elif r["pll"] < 0.05:
            ax_a.text(x6[i], max(r["rll_kg"], r["rll_llm"], 0) + 0.02, "*", ha="center", fontsize=10)
    ax_a.axhline(0.0, color="#111111", lw=0.8)
    ax_a.set_xticks(x6)
    ax_a.set_xticklabels(ll_lbls)
    ax_a.set_ylim(-0.30, 0.32)
    ax_a.set_ylabel("r, 20 LL indicators")
    ax_a.set_title("(a) Least-likely judgments", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, loc="lower left")

    # (b) Stressor domain x polarity
    b_keys = ["IW_All16", "IW_ML8", "IW_LL8", "APH_All24", "APH_ML12", "APH_LL12"]
    sub_b = mod8.loc[b_keys]
    ax_b.bar(x6 - w / 2, sub_b["r_kg"], width=w, color=C_KG)
    ax_b.bar(x6 + w / 2, sub_b["r_llm"], width=w, color=C_BS)
    for i, (_, r) in enumerate(sub_b.iterrows()):
        if r["bh_fdr_q"] < 0.05:
            ax_b.text(x6[i], max(r["r_kg"], r["r_llm"]) + 0.02, "*", ha="center", fontsize=11)
    ax_b.axvline(2.5, ls=":", color="#777777", lw=1.0)
    ax_b.axhline(0.0, color="#111111", lw=0.8)
    ax_b.text(1.0, 0.66, "IW", ha="center", fontsize=10)
    ax_b.text(4.0, 0.66, "APH", ha="center", fontsize=10)
    ax_b.set_xticks(x6)
    ax_b.set_xticklabels(["All", "ML", "LL", "All", "ML", "LL"])
    ax_b.set_ylim(-0.15, 0.72)
    ax_b.set_ylabel("r with 5×5RS")
    ax_b.set_title("(b) Stressor domain × polarity", loc="left", fontsize=11, fontweight="bold")

    # (c) Hybrid convergence (full sample & 1,000-split CV)
    c_keys = ["Pure_Base_LLM", "Pure_KG_LLM", "Hybrid_Polarity", "Hybrid_Domain", "Hybrid_Dual"]
    c_lbls = ["Base", "KG", "Pol.", "Dom.", "Dual"]
    x5 = np.arange(5)
    sub_c = d4c.loc[c_keys]
    ax_c.bar(x5 - w / 2, sub_c["r_5x5rs"], width=w, color="#4D4D4D", label="Full sample")
    cv_vals = sub_c["cv_mean"].fillna(0).values
    mask_cv = ~sub_c["cv_mean"].isna().values
    xcv = x5[mask_cv] + w / 2
    ycv = sub_c.loc[mask_cv, "cv_mean"].values
    ylo = ycv - sub_c.loc[mask_cv, "cv_lo"].values
    yhi = sub_c.loc[mask_cv, "cv_hi"].values - ycv
    ax_c.bar(xcv, ycv, width=w, yerr=[ylo, yhi], capsize=3, color="#BDBDBD", label="Held-out (1,000 splits)")
    ax_c.set_xticks(x5)
    ax_c.set_xticklabels(c_lbls)
    ax_c.set_ylim(0.0, 0.74)
    ax_c.set_ylabel("r with 5×5RS")
    ax_c.set_title("(c) Hybrid forms: convergence", loc="left", fontsize=11, fontweight="bold")
    ax_c.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper left")

    # (d) Incremental validity over 5x5RS (%)
    d_keys = c_keys + ["Scenario_DualHit"]
    d_lbls = c_lbls + ["Hit"]
    sub_d = d4c.loc[d_keys]
    x6d = np.arange(6)
    wd = 0.24
    ax_d.bar(x6d - wd, sub_d["dr2_dep"] * 100.0, width=wd, color="#B2182B", label="Depression")
    ax_d.bar(x6d, sub_d["dr2_anx"] * 100.0, width=wd, color="#EF8A62", label="Anxiety")
    ax_d.bar(x6d + wd, sub_d["dr2_str"] * 100.0, width=wd, color="#8073AC", label="Stress")
    for i, (_, r) in enumerate(sub_d.iterrows()):
        for off, col_v, col_p in [(-wd, "dr2_dep", "p_dr2_dep"), (0.0, "dr2_anx", "p_dr2_anx"), (wd, "dr2_str", "p_dr2_str")]:
            if r[col_p] < 0.01:
                ax_d.text(x6d[i] + off, r[col_v] * 100.0 + 0.08, "**", ha="center", fontsize=8.8)
            elif r[col_p] < 0.05:
                ax_d.text(x6d[i] + off, r[col_v] * 100.0 + 0.08, "*", ha="center", fontsize=8.8)
    ax_d.set_xticks(x6d)
    ax_d.set_xticklabels(d_lbls)
    ax_d.set_ylim(0.0, 4.2)
    ax_d.set_ylabel(r"$\Delta R^2$ over 5×5RS (%)")
    ax_d.set_title("(d) Incremental validity", loc="left", fontsize=11, fontweight="bold")
    ax_d.legend(frameon=False, loc="upper right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_figC1(raw_dir, res_s01: dict, out_path):
    """Figure C1: Knowledge graph degree distribution (log-log frequency and CCDF)."""
    import networkx as nx
    nodes_df = pd.read_csv(raw_dir / "kg_nodes.csv")
    edges_df = pd.read_csv(raw_dir / "kg_edges.csv")
    G = nx.DiGraph()
    for nid in nodes_df["node_id"]:
        G.add_node(nid)
    for _, r in edges_df.iterrows():
        G.add_edge(r["source_id"], r["target_id"])
    degs = np.array([d for _, d in G.degree()], dtype=int)
    vals, cnts = np.unique(degs[degs > 0], return_counts=True)
    ccdf = np.array([np.mean(degs >= k) for k in vals])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 3.8), gridspec_kw={"wspace": 0.32})
    ax_a.scatter(vals, cnts, color=C_KG, s=28)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.text(0.96, 0.94, f"n = {res_s01['n_nodes']:,} nodes\nmean degree {res_s01['mean_degree']:.2f}\nmax {res_s01['max_degree']} (resilience)\n{res_s01['leaf_pct']:.0f}% leaves",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=9.2)
    ax_a.set_xlabel("Degree k"); ax_a.set_ylabel("Number of nodes")
    ax_a.set_title("(a) Degree frequency", loc="left", fontsize=11, fontweight="bold")

    ax_b.step(vals, ccdf, where="post", color=C_KG, lw=1.8)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel("Degree k"); ax_b.set_ylabel(r"P(K $\geq$ k)")
    ax_b.set_title("(b) Cumulative distribution", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_figC2(res_s01: dict, out_path):
    """Figure C2: Top-15 PageRank concepts and Top-10 relation predicates."""
    c2 = res_s01["table_c2"].iloc[::-1]
    c3 = res_s01["table_c3"].iloc[::-1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 4.4), gridspec_kw={"wspace": 0.65})
    ya = np.arange(len(c2))
    ax_a.barh(ya, c2["pagerank"], color=C_KG, height=0.72)
    for i, v in enumerate(c2["pagerank"]):
        ax_a.text(v + 0.0007, ya[i], f"{v:.4f}", va="center", fontsize=8.8)
    ax_a.set_yticks(ya); ax_a.set_yticklabels(c2["node"])
    ax_a.set_xlim(0, 0.058)
    ax_a.set_xlabel("PageRank (directed, edge-weighted)")
    ax_a.set_title("(a) Top-15 concepts\nby PageRank", loc="left", fontsize=10.8, fontweight="bold")

    yb = np.arange(len(c3))
    ax_b.barh(yb, c3["number_of_triples"], color=C_BS, height=0.72)
    for i, v in enumerate(c3["number_of_triples"]):
        ax_b.text(v + 4, yb[i], str(int(v)), va="center", fontsize=8.8)
    ax_b.set_yticks(yb); ax_b.set_yticklabels(c3["relation_predicate"])
    ax_b.set_xlim(0, 355)
    ax_b.set_xlabel("Number of triples")
    ax_b.set_title(f"(b) Top-10 relation\npredicates (of {res_s01['n_predicates']}\ntypes)", loc="left", fontsize=10.8, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pilot_heatmap(ml_mat: np.ndarray, ll_mat: np.ndarray, cmap_name: str, out_path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.6, 10.2), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, mat, title in [
        (ax_l, ml_mat, f"Most likely\nmean modal share {ml_mat.max(axis=1).mean():.2f}"),
        (ax_r, ll_mat, f"Least likely\nmean modal share {ll_mat.max(axis=1).mean():.2f}"),
    ]:
        im = ax.imshow(mat, cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="auto")
        for r in range(30):
            for c in range(5):
                val = mat[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8.2, color="white" if val >= 0.54 else "#111111")
        ax.set_xticks(range(5)); ax.set_xticklabels(["A", "B", "C", "D", "E"])
        ax.set_xlabel("Option")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax_l.set_yticks(range(30)); ax_l.set_yticklabels([f"S{i}" for i in range(1, 31)])
    ax_l.set_ylabel("Scenario")
    cbar = fig.colorbar(im, ax=[ax_l, ax_r], orientation="horizontal", fraction=0.025, pad=0.06)
    cbar.set_label("Share of pilot respondents")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_figD1(res_s09: dict, out_path):
    _plot_pilot_heatmap(res_s09["pilot_a_ml"], res_s09["pilot_a_ll"], "Greens", out_path)


def plot_figD2(res_s09: dict, out_path):
    _plot_pilot_heatmap(res_s09["pilot_b_ml"], res_s09["pilot_b_ll"], "Blues", out_path)


def plot_figD3(res_s08: dict, out_path):
    """Figure D3: Internal consistency and criterion correlations across 4 scoring keys."""
    ak = res_s08["alternative_keys"].iloc[::-1]
    lbls = ["Modal\nbinary", "Borda\nrank", "Formal\nLOO", "Pilot\nproportion"]
    y = np.arange(4)
    h = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.6, 4.0), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, ck, cb, xl, xmax in [
        (ax1, "alpha_kg", "alpha_llm", "Cronbach's α", 1.0),
        (ax2, "r_5x5rs_kg", "r_5x5rs_llm", "r with 5×5RS", 0.50),
        (ax3, "r_pa_kg", "r_pa_llm", "r with PA", 0.50),
    ]:
        ax.barh(y + h / 2, ak[ck], height=h, color=C_KG, label="KG-LLM")
        ax.barh(y - h / 2, ak[cb], height=h, color=C_BS, label="base LLM")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xl)
    ax1.set_yticks(y); ax1.set_yticklabels(lbls)
    ax1.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.18))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_figD4(res_s08: dict, out_path):
    """Figure D4: Cross-stage associations across 80 scored indicators."""
    vecs = res_s08["cross_stage_vectors"]
    cs = res_s08["cross_stage_table"].set_index("predictor")
    is_kg = vecs["is_kg"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(9.0, 3.8), gridspec_kw={"wspace": 0.45})

    # (a) Expert option rating vs item r with 5x5RS
    x1, y1 = vecs["ex_opt"], vecs["rj_5x5"]
    ax_a.scatter(x1[is_kg], y1[is_kg], color=C_KG, alpha=0.72, s=30)
    ax_a.scatter(x1[~is_kg], y1[~is_kg], color=C_BS, alpha=0.72, s=30)
    xs = np.linspace(x1.min(), x1.max(), 40)
    m, b = np.polyfit(x1, y1, 1)
    ax_a.plot(xs, m * xs + b, color="#333333", lw=1.4)
    ax_a.axhline(0.0, color="#888888", lw=0.8)
    r_opt = cs.loc["Expert_option", "r80_5x5rs"]
    p_opt = cs.loc["Expert_option", "p80_5x5rs"]
    ax_a.text(0.04, 0.94, f"r = {r_opt:.2f}, p = {p_opt:.3f}", transform=ax_a.transAxes, va="top", fontsize=9.2)
    ax_a.set_xlabel("Expert option rating"); ax_a.set_ylabel("Item r with 5×5RS")
    ax_a.set_title("(a) Expert rating", loc="left", fontsize=10.8, fontweight="bold")

    # (b) Pilot consensus vs formal item cITC
    x2, y2 = vecs["p_max"], vecs["citc_40"]
    ax_b.scatter(x2[is_kg], y2[is_kg], color=C_KG, alpha=0.72, s=30)
    ax_b.scatter(x2[~is_kg], y2[~is_kg], color=C_BS, alpha=0.72, s=30)
    xs2 = np.linspace(x2.min(), x2.max(), 40)
    m2, b2 = np.polyfit(x2, y2, 1)
    ax_b.plot(xs2, m2 * xs2 + b2, color="#333333", lw=1.4)
    ax_b.axhline(0.0, color="#888888", lw=0.8)
    r_pm = cs.loc["Pilot_consensus", "r80_citc40"]
    ax_b.text(0.04, 0.94, f"r = {r_pm:.2f}, p < .001", transform=ax_b.transAxes, va="top", fontsize=9.2)
    ax_b.set_xlabel("Pilot modal consensus"); ax_b.set_ylabel("Formal item cITC")
    ax_b.set_title("(b) Pilot consensus", loc="left", fontsize=10.8, fontweight="bold")

    # (c) Predictors horizontal bars
    p_order = ["Expert_option", "Expert_scenario", "Pilot_entropy", "Key_stability", "Pilot_consensus"]
    p_lbls = ["Expert option", "Expert scenario", "Pilot entropy", "Key stability", "Pilot consensus"]
    sub_p = cs.loc[p_order]
    yp = np.arange(5)
    h = 0.35
    ax_c.barh(yp + h / 2, sub_p["r80_citc40"], height=h, color="#3B3B3B", label="cITC")
    ax_c.barh(yp - h / 2, sub_p["r80_5x5rs"], height=h, color="#AEAEAE", label="r with 5×5RS")
    ax_c.axvline(0.0, color="#222222", lw=0.8)
    ax_c.set_yticks(yp); ax_c.set_yticklabels(p_lbls)
    ax_c.set_xlim(-0.62, 0.68)
    ax_c.set_xlabel("Correlation (80 indicators)")
    ax_c.set_title("(c) Predictors", loc="left", fontsize=10.8, fontweight="bold")
    ax_c.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.22))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
