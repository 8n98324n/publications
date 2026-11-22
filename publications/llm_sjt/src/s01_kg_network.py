# -*- coding: utf-8 -*-
"""s01_kg_network.py: Stage 1-2 corpus audit, ReBEL-LLM overlap, NetworkX graph metrics, and ontology audit (§3.1, App. C, F.1)."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import RAW_DIR, TAB_DIR

STRICT_TERMS = [
    "cd-risc", "brs", "rsa", "scale", "item", "measure", "questionnaire", "inventory",
    "survey", "instrument", "subscale", "alpha", "cronbach", "valid", "reliab", "score",
    "data", "analysis", "sample", "study", "review", "meta-analysis", "paper", "article",
    "concept", "definition",
]
GENERAL_META_TERMS = ["factor", "model", "framework", "dimension", "construct"]
BROAD_TERMS = STRICT_TERMS + GENERAL_META_TERMS

SEED_KEYWORDS = [
    "resilience", "psychological resilience", "adaptability", "emotion regulation",
    "optimism", "self-efficacy", "social support", "coping", "adversity", "stress",
    "self regulation", "personality", "environment",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def run() -> dict:
    meta = json.loads((RAW_DIR / "corpus_stage12_metadata.json").read_text(encoding="utf-8"))
    reb = pd.read_csv(RAW_DIR / "rebel_triples_raw.csv").fillna("")
    ds = pd.read_csv(RAW_DIR / "deepseek_triples_raw.csv").fillna("")

    # DeepSeek sentence-level verification decisions over 2,593 ReBEL outputs
    sent_grp = ds.groupby("sentence_index")["conclusion"].first()
    n_rebel_nonempty = int(len(sent_grp))
    n_accepted = int((sent_grp == "correct").sum())
    n_reextracted_sents = int((sent_grp == "incorrect").sum())
    n_no_triple = int((sent_grp == "incorrect_no_triple").sum())
    n_replacement_triples = int((ds["conclusion"] == "incorrect").sum())

    # Overlap between ReBEL and raw DeepSeek outputs (§3.1.1)
    ent_a, so_a, tr_a = set(), set(), set()
    for _, r in reb.iterrows():
        s, rl, o = _norm(r["subject"]), _norm(r["relation"]), _norm(r["object"])
        if s and rl and o:
            ent_a.update([s, o]); so_a.add((s, o)); tr_a.add((s, rl, o))

    ent_b, so_b, tr_b = set(), set(), set()
    for _, r in ds.iterrows():
        s, rl, o = _norm(r["subject"]), _norm(r["relation"]), _norm(r["object"])
        if s and rl and o:
            ent_b.update([s, o]); so_b.add((s, o)); tr_b.add((s, rl, o))

    overlap_rows = []
    for level, A, B in [("entities", ent_a, ent_b), ("subject_object", so_a, so_b), ("exact_triple", tr_a, tr_b)]:
        inter = len(A & B)
        overlap_rows.append({
            "level": level,
            "rebel_unique": len(A),
            "llm_unique": len(B),
            "shared": inter,
            "jaccard": inter / len(A | B),
            "dice": 2.0 * inter / (len(A) + len(B)),
            "szymkiewicz_simpson": inter / min(len(A), len(B)),
        })
    df_overlap = pd.DataFrame(overlap_rows)
    df_overlap.to_csv(TAB_DIR / "s01_model_overlap.csv", index=False)

    # Build weighted directed graph (3,756 nodes, 4,072 directed edges from 4,265 unique triples)
    nodes_df = pd.read_csv(RAW_DIR / "kg_nodes.csv").fillna("")
    edges_df = pd.read_csv(RAW_DIR / "kg_edges.csv").fillna("")

    G = nx.DiGraph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["node_id"], name=str(r["name"]), stored_pagerank=float(r["stored_pagerank"]))
    for _, r in edges_df.iterrows():
        u, v, w = r["source_id"], r["target_id"], int(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
            G[u][v]["predicates"].append(str(r["relation"]))
        else:
            G.add_edge(u, v, weight=w, predicates=[str(r["relation"])])

    U = G.to_undirected()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = n_edges / float(n_nodes * (n_nodes - 1))
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    largest_wcc_nodes = wccs[0]
    U_lcc = U.subgraph(largest_wcc_nodes).copy()

    avg_path = float(nx.average_shortest_path_length(U_lcc))
    diameter = int(nx.diameter(U_lcc))
    assortativity = float(nx.degree_assortativity_coefficient(U))
    mean_clust = float(nx.average_clustering(U))
    transitivity = float(nx.transitivity(U))
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    btw = nx.betweenness_centrality(U)

    stored_pr_vec = np.array([G.nodes[n]["stored_pagerank"] for n in G.nodes()])
    recomp_pr_vec = np.array([pr[n] for n in G.nodes()])
    pr_spearman = float(stats.spearmanr(stored_pr_vec, recomp_pr_vec).statistic)

    # Top 15 PageRank table (Table C2)
    top15_rows = []
    for rank, (nid, pval) in enumerate(sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15], start=1):
        top15_rows.append({
            "rank": rank,
            "node_id": nid,
            "node": G.nodes[nid]["name"],
            "pagerank": pval,
            "total_degree": G.degree(nid),
            "in_degree": G.in_degree(nid),
            "out_degree": G.out_degree(nid),
            "betweenness": btw[nid],
        })
    df_top15 = pd.DataFrame(top15_rows)
    df_top15.to_csv(TAB_DIR / "table_C2_top15_pagerank.csv", index=False)

    # Top 10 relation predicates (Table C3; across 4,265 unique triples)
    pred_counts = edges_df["relation"].value_counts()
    n_pred_types = int(pred_counts.shape[0])
    df_top10_pred = pred_counts.head(10).reset_index()
    df_top10_pred.columns = ["relation_predicate", "number_of_triples"]
    df_top10_pred.insert(0, "rank", range(1, len(df_top10_pred) + 1))
    df_top10_pred.to_csv(TAB_DIR / "table_C3_top10_predicates.csv", index=False)

    res_nid = df_top15.iloc[0]["node_id"]

    # Ontology granularity audit (Strict vs Broad; Appendix F.1 & Figure 4a)
    def has_term(txt: str, terms: list[str]) -> bool:
        low = txt.lower()
        return any(t in low for t in terms)

    node_strict = {r["node_id"]: has_term(str(r["name"]), STRICT_TERMS) for _, r in nodes_df.iterrows()}
    node_broad = {r["node_id"]: has_term(str(r["name"]), BROAD_TERMS) for _, r in nodes_df.iterrows()}

    nodes_sorted_pr = sorted(nodes_df.to_dict("records"), key=lambda r: pr[r["node_id"]], reverse=True)
    seeds_30 = [
        r for r in nodes_sorted_pr
        if any(kw in str(r["name"]).lower() for kw in SEED_KEYWORDS)
    ][:30]

    strict_nodes_n = sum(node_strict.values())
    broad_nodes_n = sum(node_broad.values())
    strict_seeds_n = sum(node_strict[r["node_id"]] for r in seeds_30)
    broad_seeds_n = sum(node_broad[r["node_id"]] for r in seeds_30)
    strict_triples_n = sum(1 for _, r in edges_df.iterrows() if node_strict[r["source_id"]] or node_strict[r["target_id"]])
    broad_triples_n = sum(1 for _, r in edges_df.iterrows() if node_broad[r["source_id"]] or node_broad[r["target_id"]])

    df_c1 = pd.DataFrame([
        ("Original triples (after manual screening)", meta["screened_triples_before_dedup"]),
        ("Unique triples after frequency aggregation", len(edges_df)),
        ("Number of nodes", n_nodes),
        ("Number of directed edges (unique node pairs)", n_edges),
        ("Number of relation predicate types", n_pred_types),
        ("Graph density", round(density, 5)),
        ("Mean degree", round(float(degrees.mean()), 2)),
        ("Median degree", int(np.median(degrees))),
        ("Maximum degree (resilience)", int(degrees.max())),
        ("Proportion of leaf nodes (degree = 1)", round(float(np.mean(degrees == 1) * 100), 1)),
        ("Proportion of highly connected nodes (degree >= 10)", round(float(np.mean(degrees >= 10) * 100), 1)),
        ("Number of weakly connected components", len(wccs)),
        ("Proportion of nodes in largest component (%)", round(len(largest_wcc_nodes) / n_nodes * 100, 1)),
        ("Average shortest path length in largest component", round(avg_path, 2)),
        ("Diameter of largest component", diameter),
        ("Degree assortativity coefficient", round(assortativity, 3)),
        ("Mean clustering coefficient (undirected)", round(mean_clust, 3)),
        ("Global clustering coefficient (transitivity)", round(transitivity, 3)),
        ("Highest betweenness centrality (resilience)", round(float(btw[res_nid]), 2)),
    ], columns=["Indicator", "Value"])
    df_c1.to_csv(TAB_DIR / "table_C1_network_stats.csv", index=False)

    return {
        "corpus_meta": meta,
        "rebel_nonempty": n_rebel_nonempty,
        "deepseek_accepted": n_accepted,
        "deepseek_reextracted_sents": n_reextracted_sents,
        "deepseek_no_triple": n_no_triple,
        "deepseek_replacement_triples": n_replacement_triples,
        "deepseek_raw_rows": int(len(ds)),
        "overlap": df_overlap,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_unique_triples": int(len(edges_df)),
        "n_predicates": n_pred_types,
        "density": density,
        "mean_degree": float(degrees.mean()),
        "median_degree": int(np.median(degrees)),
        "max_degree": int(degrees.max()),
        "leaf_pct": float(np.mean(degrees == 1) * 100.0),
        "hub10_pct": float(np.mean(degrees >= 10) * 100.0),
        "n_wcc": len(wccs),
        "largest_wcc_pct": float(len(largest_wcc_nodes) / n_nodes * 100.0),
        "avg_shortest_path": avg_path,
        "diameter": diameter,
        "assortativity": assortativity,
        "mean_clustering": mean_clust,
        "transitivity": transitivity,
        "resilience_pagerank": float(pr[res_nid]),
        "resilience_betweenness": float(btw[res_nid]),
        "pagerank_spearman_with_stored": pr_spearman,
        "strict_nodes_n": strict_nodes_n,
        "strict_nodes_pct": strict_nodes_n / n_nodes * 100.0,
        "broad_nodes_n": broad_nodes_n,
        "broad_nodes_pct": broad_nodes_n / n_nodes * 100.0,
        "strict_seeds_n": strict_seeds_n,
        "strict_seeds_pct": strict_seeds_n / 30.0 * 100.0,
        "broad_seeds_n": broad_seeds_n,
        "broad_seeds_pct": broad_seeds_n / 30.0 * 100.0,
        "strict_triples_n": strict_triples_n,
        "strict_triples_pct": strict_triples_n / len(edges_df) * 100.0,
        "broad_triples_n": broad_triples_n,
        "broad_triples_pct": broad_triples_n / len(edges_df) * 100.0,
        "table_c1": df_c1,
        "table_c2": df_top15,
        "table_c3": df_top10_pred,
    }


if __name__ == "__main__":
    out = run()
    print(f"s01_kg_network complete: {out['n_nodes']} nodes, {out['n_edges']} edges, PR_resilience={out['resilience_pagerank']:.4f}")
