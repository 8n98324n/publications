# -*- coding: utf-8 -*-
"""fig_part1.py: Publication-grade 300-DPI matplotlib generators for Figures 1 through 8."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_KG = "#2CA25F"
C_KG_LIGHT = "#8FD1A8"
C_KG_DARK = "#145A32"
C_BS = "#2B7BBA"
C_BS_LIGHT = "#8EC0E4"
C_REB = "#D95F0E"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})


def plot_fig01(res_s01: dict, out_path):
    """Figure 1: Four stages of SJT development and validation, mapped to H1-H4."""
    meta = res_s01["corpus_meta"]
    fig, ax = plt.subplots(figsize=(8.2, 10.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.5)
    ax.axis("off")

    bands = [
        (10.35, 2.95, "#EEF4FA", "Stage 1\nKnowledge base"),
        (6.95, 3.15, "#F6F5F0", "Stage 2\nKG construction"),
        (3.45, 3.25, "#EEF4FA", "Stage 3\nItem generation"),
        (0.20, 3.00, "#F6F5F0", "Stage 4\nEvaluation"),
    ]
    for y0, h, fc, lbl in bands:
        ax.add_patch(FancyBboxPatch((0.15, y0), 9.7, h, boxstyle="round,pad=0.08,rounding_size=0.18", facecolor=fc, edgecolor="none"))
        ax.text(0.45, y0 + h / 2, lbl, rotation=90, va="center", ha="center", fontsize=10.5, fontweight="bold", color="#3A3A3A")

    def box(x, y, w, h, txt, fc="#FFFFFF", ec="#4A4A4A", lw=1.5, fs=9.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14", facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(x + w / 2, y + h / 2, txt, va="center", ha="center", fontsize=fs, color="#111111", linespacing=1.28)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#4A4A4A", lw=1.5, mutation_scale=13))

    # Stage 1
    n_rev = meta["corpus_pdfs"]["conceptual_reviews_count"]
    n_scl = meta["corpus_pdfs"]["measurement_scales_count"]
    n_cand = meta["rag_candidates_top6000"]["total"]
    n_clean = meta["semantic_cleaned_sentences"]["total"]
    box(0.95, 11.95, 3.6, 1.05, f"{n_rev + n_scl} resilience papers (PDF)\n{n_rev} theory, {n_scl} measurement")
    box(0.95, 10.60, 3.6, 0.85, "Split, embed, vector store")
    box(5.15, 10.60, 4.4, 2.40, f"RAG retrieval:\ntop {n_cand:,} sentences\nSemantic cleaning:\n{n_clean:,} retained", fc="#EAF6EC", ec="#4A4A4A")
    arrow(2.75, 11.95, 2.75, 11.45)
    arrow(4.55, 11.02, 5.15, 11.02)

    # Stage 2
    box(0.95, 8.70, 3.6, 1.15, f"ReBEL extraction\n{res_s01['rebel_nonempty']:,} triples", fc="#FDE8D7", ec=C_REB)
    box(5.15, 8.95, 4.4, 1.00, f"DeepSeek-Chat extraction\n{res_s01['deepseek_raw_rows']:,} triples", fc="#E3EFFB", ec=C_BS)
    box(5.15, 7.92, 4.4, 0.72, f"Screening: {meta['screened_triples_before_dedup']:,} triples", fc="#E3EFFB", ec=C_BS)
    box(5.15, 7.08, 4.4, 0.65, f"KG: {res_s01['n_nodes']:,} nodes, {res_s01['n_edges']:,} edges", fc="#E5F6EA", ec=C_KG)
    ax.plot([7.35, 7.35], [10.60, 10.30], color="#4A4A4A", lw=1.5)
    ax.plot([2.75, 7.35], [10.30, 10.30], color="#4A4A4A", lw=1.5)
    arrow(2.75, 10.30, 2.75, 9.85)
    arrow(7.35, 10.30, 7.35, 9.95)
    arrow(7.35, 8.95, 7.35, 8.64)
    arrow(7.35, 7.92, 7.35, 7.73)
    ax.text(2.75, 8.28, "H2: triple quality", ha="center", va="center", fontsize=10.5, color=C_REB, fontweight="bold")
    ax.text(2.75, 7.50, "Graph content audit", ha="center", va="center", fontsize=9.8, color="#8C2D19")

    # Stage 3
    box(0.95, 5.15, 3.0, 1.15, "Path A: base LLM\n30 items (5 × 6)", fc="#E3EFFB", ec=C_BS)
    box(6.05, 5.15, 3.5, 1.15, "Path B: KG-LLM\n30 items (cited)", fc="#E5F6EA", ec=C_KG)
    ax.text(5.00, 5.92, "H3: item\nquality", ha="center", va="center", fontsize=10.5, color=C_KG, fontweight="bold")
    ax.text(5.00, 5.25, "Citation count", ha="center", va="center", fontsize=9.6, color="#8C2D19")
    arrow(7.80, 7.08, 7.80, 6.30)
    box(0.95, 3.62, 8.6, 1.15, "Each item: second-person scenario (~100 characters)\nand five options: adaptability, emotion regulation,\noptimism, self-efficacy, social support", fc="#FAF9F6")
    arrow(2.45, 5.15, 2.45, 4.77)
    arrow(7.80, 5.15, 7.80, 4.77)

    # Stage 4
    box(0.85, 1.40, 2.9, 1.30, "Expert evaluation\n5 experts, 60 items\n20 retained per path")
    box(4.05, 1.40, 2.25, 1.30, "Pilot tests\nN = 50 / 52\nScoring keys")
    box(6.60, 1.40, 3.0, 1.30, "Formal test, N = 150\n5×5RS, PANAS,\nDASS-21")
    ax.plot([5.25, 5.25], [3.62, 3.12], color="#4A4A4A", lw=1.5)
    ax.plot([2.30, 5.25], [3.12, 3.12], color="#4A4A4A", lw=1.5)
    arrow(2.30, 3.12, 2.30, 2.70)
    arrow(3.75, 2.05, 4.05, 2.05)
    arrow(6.30, 2.05, 6.60, 2.05)
    ax.text(5.25, 0.86, "H1: reliability and validity     H4: gain from the KG", ha="center", va="center", fontsize=10.5, color="#333333")
    ax.text(5.25, 0.36, "Exploratory: expert ratings and item functioning, judgment polarity\nand stressor domain, hybrid test assembly", ha="center", va="center", fontsize=9.4, color="#512DA8")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig02(res_s02: dict, out_path):
    """Figure 2: Expert ratings of ReBEL and LLM triples across 100 source sentences."""
    agr = res_s02["agreement"]
    comp = res_s02["paired_tests"]
    dims = ["correctness", "relevance", "completeness", "clarity_conciseness"]
    short_lbls = ["Corr.", "Relev.", "Compl.", "Clarity"]
    full_lbls = ["Correctness", "Relevance", "Completeness", "Clarity"]

    fig = plt.figure(figsize=(8.6, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.45, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, :])

    x = np.arange(4)
    for ax, metric, lo_c, hi_c, title in [
        (ax_a, "icc3k", "icc_ci_low", "icc_ci_high", "(a) ICC(3, k), F-based 95% CI"),
        (ax_b, "ac2", "ac2_ci_low", "ac2_ci_high", "(b) Gwet's AC2, bootstrap 95% CI"),
    ]:
        for offset, mname, col, lbl in [(-0.14, "ReBEL", C_REB, "ReBEL"), (0.14, "LLM", C_BS, "LLM")]:
            sub = agr[agr["model"] == mname].set_index("dimension").loc[dims]
            y = sub[metric].values
            ylo = y - sub[lo_c].values
            yhi = sub[hi_c].values - y
            ax.errorbar(x + offset, y, yerr=[ylo, yhi], fmt="o", color=col, capsize=4, markersize=7, lw=1.6, label=lbl)
        ax.axhline(0.75, ls="--", color="#777777", lw=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(short_lbls)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Agreement")
    ax_a.legend(frameon=False, ncol=2, loc="lower left")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    sub_c = comp.set_index("dimension").loc[dims]
    w = 0.35
    b1 = ax_c.bar(x - w / 2, sub_c["rebel_mean"], width=w, yerr=sub_c["rebel_sd"], capsize=4, color=C_REB, label="ReBEL")
    b2 = ax_c.bar(x + w / 2, sub_c["llm_mean"], width=w, yerr=sub_c["llm_sd"], capsize=4, color=C_BS, label="LLM")
    for i in range(4):
        ax_c.text(x[i] - w / 2, sub_c["rebel_mean"].iloc[i] + sub_c["rebel_sd"].iloc[i] + 0.12, f"{sub_c['rebel_mean'].iloc[i]:.2f}", ha="center", fontsize=9.2, color=C_REB)
        ax_c.text(x[i] + w / 2, sub_c["llm_mean"].iloc[i] + sub_c["llm_sd"].iloc[i] + 0.12, f"{sub_c['llm_mean'].iloc[i]:.2f}", ha="center", fontsize=9.2, color=C_BS)
        ax_c.text(x[i], 5.45, f"d = {sub_c['cohen_dz'].iloc[i]:.2f}\np < .001", ha="center", va="center", fontsize=9.2)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(full_lbls)
    ax_c.set_ylim(1.0, 6.25)
    ax_c.set_ylabel("Mean rating (± SD)")
    ax_c.set_title("(c) Expert quality ratings, 100 source sentences", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig03(raw_dir, out_path):
    """Figure 3: Largest connected component (88 nodes) among 120 highest-PageRank concepts."""
    nodes_df = pd.read_csv(raw_dir / "kg_nodes.csv").fillna("")
    edges_df = pd.read_csv(raw_dir / "kg_edges.csv").fillna("")
    G = nx.DiGraph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["node_id"], name=str(r["name"]))
    for _, r in edges_df.iterrows():
        u, v, w = r["source_id"], r["target_id"], int(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    top120 = [n for n, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:120]]
    S = G.subgraph(top120).copy()
    lcc_nodes = max(nx.weakly_connected_components(S), key=len)
    H = S.subgraph(lcc_nodes).copy()

    comms = list(nx.community.greedy_modularity_communities(H.to_undirected()))
    palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#BCBD22", "#17BECF"]
    node_col = {}
    for ci, cset in enumerate(comms):
        for n in cset:
            node_col[n] = palette[ci % len(palette)]

    pos = nx.spring_layout(H.to_undirected(), k=0.62, iterations=120, seed=20240501)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.axis("off")

    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#BDBDBD", alpha=0.65, arrows=True, arrowsize=9, width=0.9)
    sizes = [2200 * (pr[n] / max(pr.values())) + 55 for n in H.nodes()]
    cols = [node_col[n] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color=cols, edgecolors="white", linewidths=0.8)

    key_labels = {
        "resilience", "psychological resilience", "CD-RISC", "stress", "Russian immigrants",
        "depression", "resilience measures", "resilience factor", "protective factor",
        "protective factors", "positive adaptation", "cognitive flexibility",
        "recover from stress", "active coping", "resilient functioning", "mental health outcomes",
    }
    texts = []
    for n in H.nodes():
        nm = H.nodes[n]["name"]
        if nm in key_labels:
            x, y = pos[n]
            t = ax.text(x, y + 0.03, nm, fontsize=9.3, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78))
            texts.append(t)
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6))
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig04(res_s01: dict, res_s03: dict, res_s04: dict, raw_dir, out_path):
    """Figure 4: KG content audit, citation counts vs ratings, and rationale citations vs option rating gaps."""
    fig = plt.figure(figsize=(8.8, 10.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.05, 0.95], hspace=0.52, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    # (a) Stacked horizontal bars
    cats = [f"Triples ({res_s01['n_unique_triples']:,})", "Top-30 seeds", f"Nodes ({res_s01['n_nodes']:,})"]
    strict_pcts = np.array([res_s01["strict_triples_pct"], res_s01["strict_seeds_pct"], res_s01["strict_nodes_pct"]])
    broad_pcts = np.array([res_s01["broad_triples_pct"], res_s01["broad_seeds_pct"], res_s01["broad_nodes_pct"]])
    gen_pcts = broad_pcts - strict_pcts
    sub_pcts = 100.0 - broad_pcts

    y = np.arange(3)
    ax_a.barh(y, sub_pcts, color="#459B6C", height=0.56, label="Substantive mechanisms")
    ax_a.barh(y, gen_pcts, left=sub_pcts, color="#DFB443", height=0.56, label="General labels")
    ax_a.barh(y, strict_pcts, left=sub_pcts + gen_pcts, color="#AD1D2A", height=0.56, label="Measures and studies")
    for i in range(3):
        ax_a.text(sub_pcts[i] / 2, y[i], f"{sub_pcts[i]:.0f}%", va="center", ha="center", color="white", fontsize=9.5)
        ax_a.text(sub_pcts[i] + gen_pcts[i] / 2, y[i], f"{gen_pcts[i]:.0f}%", va="center", ha="center", color="#222222", fontsize=9.2)
        ax_a.text(sub_pcts[i] + gen_pcts[i] + strict_pcts[i] / 2, y[i], f"{strict_pcts[i]:.0f}%", va="center", ha="center", color="white", fontsize=9.5)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(cats)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Share of graph elements (%)")
    ax_a.set_title("(a) Content of the knowledge graph", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    # (b) Citations and ratings
    scen_texts = pd.read_csv(raw_dir / "scenario_texts.csv")
    wide = pd.read_csv(raw_dir / "expert_scenario_ratings.csv")
    agg = pd.read_csv(raw_dir / "item_level_expert_aggregates.csv")
    kg_t = scen_texts[scen_texts["group"] == "KG-LLM"].sort_values("within_group_id")
    nt = np.array([len(re.findall(r"\d+", str(s))) for s in kg_t["evidence_triples"]], dtype=float)
    kg_all = agg[agg["group"] == "KG-LLM"].sort_values("within_group_id")["scenario_mean"].values
    bs_all = agg[agg["group"] == "base_LLM"].sort_values("within_group_id")["scenario_mean"].values
    kg_ar = wide[[f"item{i}_arousal" for i in range(1, 31)]].values.T.mean(axis=1)

    rng = np.random.default_rng(20240501)
    jit = rng.uniform(-0.10, 0.10, size=len(nt))
    ax_b.scatter(nt + jit, kg_all, color=C_KG, s=26, label="All")
    ax_b.scatter(nt - jit, kg_ar, facecolors="white", edgecolors=C_KG, s=30, lw=1.2, label="Arousal")
    xs = np.linspace(3, 7, 50)
    m1, b1 = np.polyfit(nt, kg_all, 1)
    m2, b2 = np.polyfit(nt, kg_ar, 1)
    ax_b.plot(xs, m1 * xs + b1, color=C_KG_DARK, lw=1.6)
    ax_b.plot(xs, m2 * xs + b2, color=C_KG_DARK, lw=1.4, ls="--")
    ax_b.axhline(bs_all.mean(), color=C_BS, ls=":", lw=1.6, label="base LLM")
    ax_b.set_xticks([3, 4, 5, 6, 7])
    ax_b.set_xlabel("Cited triples per scenario")
    ax_b.set_ylabel("Mean expert rating")
    ax_b.set_title("(b) Citations and ratings", loc="left", fontsize=11, fontweight="bold")
    ax_b.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper right", handletextpad=0.3, columnspacing=0.6)

    # (c) Ratings by citations
    dims = ["realism", "relevance", "typicality", "arousal"]
    x_c = np.arange(5)
    m_3 = nt == 3
    m_57 = nt >= 5
    bs_means, bs_se, k3_means, k3_se, k5_means, k5_se = [], [], [], [], [], []
    for d in dims:
        xk = wide[[f"item{i}_{d}" for i in range(1, 31)]].values.T.mean(axis=1)
        xb = wide[[f"item{i}_{d}" for i in range(31, 61)]].values.T.mean(axis=1)
        bs_means.append(xb.mean()); bs_se.append(xb.std(ddof=1) / np.sqrt(len(xb)))
        k3_means.append(xk[m_3].mean()); k3_se.append(xk[m_3].std(ddof=1) / np.sqrt(m_3.sum()))
        k5_means.append(xk[m_57].mean()); k5_se.append(xk[m_57].std(ddof=1) / np.sqrt(m_57.sum()))
    bs_means.append(bs_all.mean()); bs_se.append(bs_all.std(ddof=1) / np.sqrt(30))
    k3_means.append(kg_all[m_3].mean()); k3_se.append(kg_all[m_3].std(ddof=1) / np.sqrt(m_3.sum()))
    k5_means.append(kg_all[m_57].mean()); k5_se.append(kg_all[m_57].std(ddof=1) / np.sqrt(m_57.sum()))

    w = 0.25
    ax_c.bar(x_c - w, bs_means, width=w, yerr=bs_se, capsize=2.5, color=C_BS, label="base LLM (n = 30)")
    ax_c.bar(x_c, k3_means, width=w, yerr=k3_se, capsize=2.5, color="#6CC26F", label=f"KG-LLM, 3 (n = {m_3.sum()})")
    ax_c.bar(x_c + w, k5_means, width=w, yerr=k5_se, capsize=2.5, color=C_KG_DARK, label=f"KG-LLM, 5–7 (n = {m_57.sum()})")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(["Re", "Rv", "Ty", "Ar", "All"])
    ax_c.set_ylim(3.4, 5.05)
    ax_c.set_ylabel("Mean rating (± SE)")
    ax_c.set_title("(c) Ratings by citations", loc="left", fontsize=11, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=8.5, loc="upper left")

    # (d) Rationale citations vs option rating gaps
    d2b = res_s04["option_table_d2b"]
    cit_pct = res_s04["rationale_citation_pct"]
    pretty_map = {
        "Adaptability": "Adaptability",
        "Emotion_Regulation": "Emotion regulation",
        "Optimism": "Optimism",
        "Self_Efficacy": "Self-efficacy",
        "Social_Support": "Social support",
    }
    for s_key, s_lbl in pretty_map.items():
        sub_s = d2b[d2b["strategy"] == s_key]
        mean_abs_g = float(np.mean(np.abs(sub_s["hedges_g"])))
        sig = bool(np.any(sub_s["welch_p"] < 0.05))
        xp = cit_pct[s_key]
        if sig:
            ax_d.scatter(xp, mean_abs_g, s=85, color="#B2182B", zorder=3)
        else:
            ax_d.scatter(xp, mean_abs_g, s=85, facecolors="white", edgecolors="#B2182B", lw=1.8, zorder=3)
        dy = 0.03 if s_key == "Self_Efficacy" else (-0.04 if s_key == "Optimism" else 0.0)
        ax_d.text(xp + 1.8, mean_abs_g + dy, s_lbl, va="center", fontsize=9.5)
    ax_d.scatter([], [], s=65, color="#B2182B", label="p < .05 on a dimension")
    ax_d.scatter([], [], s=65, facecolors="white", edgecolors="#B2182B", lw=1.8, label="not significant")
    ax_d.set_xlim(15, 100)
    ax_d.set_ylim(0.0, 0.92)
    ax_d.set_xlabel("Rationales citing a triple for the strategy (%)")
    ax_d.set_ylabel("Mean |g|, option ratings")
    ax_d.set_title("(d) Rationale citations and option rating gaps", loc="left", fontsize=11, fontweight="bold")
    ax_d.legend(frameon=False, loc="upper right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig05(res_s03: dict, out_path):
    """Figure 5: Scenario ICC(3,k) and mean expert ratings (30 scenarios per condition)."""
    st = res_s03["scenario_table"]
    lbls = ["Realism", "Relevance", "Typicality", "Arousal"]
    x = np.arange(4)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8.0, 7.2), gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.42})
    ax_a.errorbar(x - 0.14, st["icc_kg"], yerr=[st["icc_kg"] - st["icc_kg_lo"], st["icc_kg_hi"] - st["icc_kg"]],
                  fmt="o", color=C_KG, capsize=4, markersize=7.5, lw=1.6, label="KG-LLM")
    ax_a.errorbar(x + 0.14, st["icc_base"], yerr=[st["icc_base"] - st["icc_base_lo"], st["icc_base_hi"] - st["icc_base"]],
                  fmt="o", color=C_BS, capsize=4, markersize=7.5, lw=1.6, label="base LLM")
    for i in range(4):
        ax_a.text(x[i] - 0.22, st["icc_kg"].iloc[i], f"{st['icc_kg'].iloc[i]:.2f}", ha="right", va="center", color=C_KG, fontsize=9.2)
        ax_a.text(x[i] + 0.22, st["icc_base"].iloc[i], f"{st['icc_base'].iloc[i]:.2f}", ha="left", va="center", color=C_BS, fontsize=9.2)
    ax_a.axhline(0.0, color="#222222", lw=0.9)
    ax_a.axhline(0.75, color="#777777", ls="--", lw=1.1)
    ax_a.text(3.48, 0.78, ".75", color="#666666", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(lbls)
    ax_a.set_ylim(-1.5, 1.02)
    ax_a.set_ylabel("ICC(3, k), 95% CI")
    ax_a.set_title("(a) Rating consistency; typicality bootstrap p = 0.001", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="lower center")

    w = 0.34
    ax_b.bar(x - w / 2, st["kg_mean"], width=w, yerr=st["kg_sd"], capsize=4, color=C_KG)
    ax_b.bar(x + w / 2, st["base_mean"], width=w, yerr=st["base_sd"], capsize=4, color=C_BS)
    for i in range(4):
        ax_b.text(x[i] - w / 2, 1.15, f"{st['kg_mean'].iloc[i]:.2f}", ha="center", color="white", fontweight="bold", fontsize=9.2)
        ax_b.text(x[i] + w / 2, 1.15, f"{st['base_mean'].iloc[i]:.2f}", ha="center", color="white", fontweight="bold", fontsize=9.2)
        ax_b.text(x[i], 4.92, f"g = {st['hedges_g'].iloc[i]:.2f}\np = {st['welch_p'].iloc[i]:.3f}", ha="center", va="bottom", fontsize=9.0)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(lbls)
    ax_b.set_ylim(1.0, 5.85)
    ax_b.set_ylabel("Mean rating (± SD)")
    ax_b.set_title("(b) Mean expert ratings, 30 scenarios per condition", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig06(res_s04: dict, res_s05: pd.DataFrame, res_s08: dict, raw_dir, out_path):
    """Figure 6: S-CVI/Ave, rater-error MS, cross-stage correlations, item-level ratings, and 60-item screening."""
    fig = plt.figure(figsize=(8.8, 11.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.05, 0.95], hspace=0.48, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    # (a) S-CVI/Ave
    scvi = res_s04["scvi_table"]
    x5 = np.arange(5)
    w = 0.36
    ax_a.bar(x5 - w / 2, scvi["scvi_ave_kg"], width=w, color=C_KG, label="KG-LLM")
    ax_a.bar(x5 + w / 2, scvi["scvi_ave_base"], width=w, color=C_BS, label="base LLM")
    ax_a.axhline(0.90, ls="--", color="#666666", lw=1.1)
    ax_a.set_xticks(x5)
    ax_a.set_xticklabels(["Real.", "Relev.", "Typic.", "Arous.", "All"])
    ax_a.set_ylim(0.50, 1.02)
    ax_a.set_ylabel("S-CVI/Ave")
    ax_a.set_title("(a) Content validity index", loc="left", fontsize=10.8, fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(1.1, 1.24))

    # (b) MS_Error
    x4 = np.arange(4)
    ax_b.bar(x4 - w / 2, res_s05["ms_error_kg"], width=w, color=C_KG)
    ax_b.bar(x4 + w / 2, res_s05["ms_error_llm"], width=w, color=C_BS)
    ax_b.set_xticks(x4)
    ax_b.set_xticklabels(["Real.", "Relev.", "Typic.", "Arous."])
    ax_b.set_ylim(0.0, 0.72)
    ax_b.set_ylabel(r"MS$_{error}$")
    ax_b.set_title("(b) Rater-error mean square", loc="left", fontsize=10.8, fontweight="bold")

    # (c) Cross-stage predictors across 80 indicators
    cs = res_s08["cross_stage_table"].set_index("predictor").loc[
        ["Pilot_consensus", "Key_stability", "Pilot_entropy", "Option_length", "Expert_scenario", "Expert_option"]
    ]
    x6 = np.arange(6)
    wb = 0.25
    ax_c.bar(x6 - wb, cs["r80_citc40"], width=wb, color="#262626", label="cITC, 40 indicators")
    ax_c.bar(x6, cs["r80_citc20"], width=wb, color="#969696", label="cITC within polarity")
    ax_c.bar(x6 + wb, cs["r80_5x5rs"], width=wb, color="#D95F0E", label="Item r with 5×5RS")
    ax_c.axhline(0.0, color="#222222", lw=0.8)
    ax_c.axhline(0.22, ls=":", color="#777777", lw=1.0)
    ax_c.axhline(-0.22, ls=":", color="#777777", lw=1.0)
    ax_c.set_xticks(x6)
    ax_c.set_xticklabels(["Pilot\nconsensus", "Key\nstability", "Pilot\nentropy", "Option\nlength", "Expert\nscenario", "Expert\noption"])
    ax_c.set_ylim(-0.62, 0.72)
    ax_c.set_ylabel("Pearson r (80 indicators)")
    ax_c.set_title("(c) Item properties and formal item functioning", loc="left", fontsize=10.8, fontweight="bold")
    ax_c.legend(frameon=False, ncol=3, loc="upper left")

    # (d) Item-level expert ratings
    w_s, w_o, w_c = res_s04["wt_scen_60"], res_s04["wt_opt_60"], res_s04["wt_comp_60"]
    x3 = np.arange(3)
    ax_d.bar(x3 - w / 2, [w_s["m1"], w_o["m1"], w_c["m1"]], width=w, yerr=[w_s["sd1"], w_o["sd1"], w_c["sd1"]], capsize=3, color=C_KG)
    ax_d.bar(x3 + w / 2, [w_s["m2"], w_o["m2"], w_c["m2"]], width=w, yerr=[w_s["sd2"], w_o["sd2"], w_c["sd2"]], capsize=3, color=C_BS)
    for i, wt_obj in enumerate([w_s, w_o, w_c]):
        ax_d.text(x3[i], max(wt_obj["m1"] + wt_obj["sd1"], wt_obj["m2"] + wt_obj["sd2"]) + 0.06, f"g = {wt_obj['hedges_g']:.2f}", ha="center", fontsize=9.0)
    ax_d.set_xticks(x3)
    ax_d.set_xticklabels(["Scenario", "Option", "Composite"])
    ax_d.set_ylim(3.6, 4.88)
    ax_d.set_ylabel("Mean rating (± SD)")
    ax_d.set_title("(d) Item-level expert ratings", loc="left", fontsize=10.8, fontweight="bold")

    # (e) Screening of 60 items
    agg = pd.read_csv(raw_dir / "item_level_expert_aggregates.csv")
    for grp, col in [("KG-LLM", C_KG), ("base_LLM", C_BS)]:
        sub_r = agg[(agg["group"] == grp) & (agg["retained"] == 1)]
        sub_e = agg[(agg["group"] == grp) & (agg["retained"] == 0)]
        ax_e.scatter(sub_r["Total_Mean"], sub_r["Total_Std"], color=col, s=36)
        ax_e.scatter(sub_e["Total_Mean"], sub_e["Total_Std"], facecolors="white", edgecolors=col, lw=1.4, s=36)
    ax_e.scatter([], [], color="#666666", s=36, label="Retained")
    ax_e.scatter([], [], facecolors="white", edgecolors="#666666", lw=1.4, s=36, label="Excluded")
    ax_e.set_xlabel("Item mean rating")
    ax_e.set_ylabel("Item rating SD")
    ax_e.set_title("(e) Screening of 60 items", loc="left", fontsize=10.8, fontweight="bold")
    ax_e.legend(frameon=False, loc="upper right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig07(res_s04: dict, out_path):
    """Figure 7: Screening selectivity, option ratings by strategy, and rater-error MS across 10 cells."""
    fig = plt.figure(figsize=(8.6, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], width_ratios=[1.15, 0.95], hspace=0.45, wspace=0.14)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)

    # (a) Screening slope plot
    pairs = [
        (0, res_s04["wt_kg_ret_scen"], C_KG, "KG-LLM\nscenario"),
        (1, res_s04["wt_kg_ret_opt"], C_KG, "KG-LLM\noption"),
        (2.2, res_s04["wt_bs_ret_scen"], C_BS, "base LLM\nscenario"),
        (3.2, res_s04["wt_bs_ret_opt"], C_BS, "base LLM\noption"),
    ]
    xticks, xlbls = [], []
    for x0, wt, col, lbl in pairs:
        e_m, r_m = wt["m2"], wt["m1"]
        ax_a.plot([x0 - 0.16, x0 + 0.16], [e_m, r_m], color=col, lw=1.8)
        ax_a.scatter([x0 - 0.16], [e_m], facecolors="white", edgecolors=col, s=55, lw=1.8, zorder=3)
        ax_a.scatter([x0 + 0.16], [r_m], color=col, s=55, zorder=3)
        ns_tag = "\nn.s." if wt["p"] >= 0.05 else ""
        ax_a.text(x0 + 0.25, 0.5 * (e_m + r_m), f"g = {wt['hedges_g']:.2f}{ns_tag}", va="center", color=col, fontsize=9.2)
        xticks.append(x0); xlbls.append(lbl)
    ax_a.scatter([], [], facecolors="white", edgecolors="#666666", s=50, lw=1.6, label="Eliminated (10)")
    ax_a.scatter([], [], color="#666666", s=50, label="Retained (20)")
    ax_a.set_xticks(xticks)
    ax_a.set_xticklabels(xlbls)
    ax_a.set_xlim(-0.5, 4.1)
    ax_a.set_ylim(3.85, 4.55)
    ax_a.set_ylabel("Mean expert rating")
    ax_a.set_title("(a) Screening effect", loc="left", fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="upper left")

    # (b) & (c) Option ratings and rater error
    d2b = res_s04["option_table_d2b"]
    strats = ["Adaptability", "Emotion_Regulation", "Optimism", "Self_Efficacy", "Social_Support"]
    pretty = ["Adaptability", "Emotion regulation", "Optimism", "Self-efficacy", "Social support"]
    y_pos = np.arange(5)[::-1]

    for idx, st_name in enumerate(strats):
        yb = y_pos[idx]
        rt = d2b[(d2b["strategy"] == st_name) & (d2b["dimension"] == "Typicality")].iloc[0]
        rd = d2b[(d2b["strategy"] == st_name) & (d2b["dimension"] == "Distinctiveness")].iloc[0]

        ax_b.plot([rt["kg_mean"], rt["base_mean"]], [yb + 0.14, yb + 0.14], color="#BBBBBB", lw=1.6, zorder=1)
        ax_b.scatter([rt["kg_mean"]], [yb + 0.14], color=C_KG, marker="o", s=52, zorder=3)
        ax_b.scatter([rt["base_mean"]], [yb + 0.14], color=C_BS, marker="o", s=52, zorder=3)
        if rt["welch_p"] < 0.05:
            ax_b.text(max(rt["kg_mean"], rt["base_mean"]) + 0.015, yb + 0.14, "*", va="center", fontsize=11)

        ax_b.plot([rd["kg_mean"], rd["base_mean"]], [yb - 0.14, yb - 0.14], color="#BBBBBB", lw=1.6, zorder=1)
        ax_b.scatter([rd["kg_mean"]], [yb - 0.14], color=C_KG, marker="s", s=48, zorder=3)
        ax_b.scatter([rd["base_mean"]], [yb - 0.14], color=C_BS, marker="s", s=48, zorder=3)
        if rd["welch_p"] < 0.05:
            ax_b.text(max(rd["kg_mean"], rd["base_mean"]) + 0.015, yb - 0.14, "*", va="center", fontsize=11)

        h = 0.16
        ax_c.barh(yb + 0.24, rt["ms_error_kg"], height=h, color=C_KG)
        ax_c.barh(yb + 0.08, rt["ms_error_base"], height=h, color=C_BS)
        ax_c.barh(yb - 0.08, rd["ms_error_kg"], height=h, color=C_KG_LIGHT)
        ax_c.barh(yb - 0.24, rd["ms_error_base"], height=h, color=C_BS_LIGHT)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(pretty)
    ax_b.set_xlim(4.15, 4.62)
    ax_b.set_xlabel("Mean option rating")
    ax_b.set_title("(b) Option ratings", loc="left", fontsize=11, fontweight="bold")

    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_c.set_xlim(0.0, 0.36)
    ax_c.set_xlabel(r"MS$_{error}$")
    ax_c.set_title("(c) Rater error", loc="left", fontsize=11, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig08(res_s06: dict, out_path):
    """Figure 8: Cross-classified mixed-effects model variance components and marginal means."""
    sh = res_s06["variance_shares"]
    mm = res_s06["marginal_means"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 4.3), gridspec_kw={"width_ratios": [0.95, 1.15], "wspace": 0.36})
    comps = ["Theme", "Scenario", "Rater", "Residual"]
    vals = [sh["theme_pct"], sh["item_pct"], sh["rater_pct"], sh["residual_pct"]]
    cols = ["#9ECAE1", "#525252", "#8C6BB1", "#CBD2D9"]
    y = np.arange(4)
    ax_a.barh(y, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax_a.text(v + 1.5, y[i], f"{v:.1f}%", va="center", fontsize=9.5)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(comps)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("% of rating variance")
    ax_a.set_title("(a) Variance components", loc="left", fontsize=11, fontweight="bold")

    dims = ["realism", "relevance", "typicality", "arousal"]
    x = np.arange(4)
    mk = mm[mm["condition"] == "KG-LLM"].set_index("dimension").loc[dims]
    mb = mm[mm["condition"] == "base_LLM"].set_index("dimension").loc[dims]
    ax_b.errorbar(x - 0.10, mk["marginal_mean"], yerr=mk["se"], fmt="o", color=C_KG, capsize=4, markersize=8, lw=1.6, label="KG-LLM")
    ax_b.errorbar(x + 0.10, mb["marginal_mean"], yerr=mb["se"], fmt="s", color=C_BS, capsize=4, markersize=8, lw=1.6, label="base LLM")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Real.", "Relev.", "Typic.", "Arous."])
    ax_b.set_ylim(3.80, 4.60)
    ax_b.set_ylabel("Expert rating, mean ± SE")
    ax_b.set_title("(b) Condition effect by\ndimension", loc="left", fontsize=11, fontweight="bold")
    ax_b.legend(frameon=False, ncol=2, loc="lower center")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
