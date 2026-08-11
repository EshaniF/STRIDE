"""
temporal_difficulty_analysis.py

Tests the assumption that "later snapshots are harder" by correlating
per-snapshot model performance (MRR) with temporal position.

This mirrors the structure of `test()` in train_curriculum.py / main.py,
but additionally records per-snapshot MRR and produces:
    1. A CSV of (snapshot_index, mrr_filter, mrr_raw, num_triples)
    2. A Spearman correlation between snapshot_index and (1 - mrr_filter)
    3. A publication-ready scatter + trend plot

Usage:
    from temporal_difficulty_analysis import analyze_temporal_difficulty

    results_df, corr_stats = analyze_temporal_difficulty(
        model, train_list, test_list, num_rels, num_nodes,
        use_cuda, all_ans_list_test, static_graph, args,
        save_prefix="ICEWS14_temporal_difficulty"
    )
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from rgcn.utils import build_sub_graph, get_total_rank
from rgcn import utils


def analyze_temporal_difficulty(
    model,
    history_list,
    test_list,
    num_rels,
    num_nodes,
    use_cuda,
    all_ans_list,
    static_graph,
    args,
    save_prefix="temporal_difficulty",
    eval_bz=1000,
):
    """
    Runs evaluation snapshot-by-snapshot (same protocol as test()) and
    records per-snapshot MRR, then correlates it with temporal position.

    Returns:
        results_df: pandas DataFrame with one row per test snapshot
        corr_stats: dict with spearman correlation, p-value, and summary text
    """
    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    per_snapshot_records = []

    with torch.no_grad():
        for time_idx, test_snap in enumerate(tqdm(test_list, desc="Evaluating per-snapshot MRR")):
            history_glist = [
                build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                for g in input_list
            ]
            test_triples_input = (
                torch.LongTensor(test_snap).cuda()
                if use_cuda
                else torch.LongTensor(test_snap)
            )
            test_triples_input = test_triples_input.to(args.gpu) if use_cuda else test_triples_input

            # NOTE: adjust this call to match your model's predict() signature.
            # This assumes the RE-GCN-style predict(test_graph, num_rels, static_graph, test_triplets, use_cuda)
            # -> (all_triples, score, score_rel). If using the LogCL/STRIDE model's predict(),
            # swap in that signature instead (see train_curriculum.py's test()).
            test_triples, final_score, _ = model.predict(
                history_glist, num_rels, static_graph, test_triples_input, use_cuda
            )

            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = get_total_rank(
                test_triples, final_score, all_ans_list[time_idx], eval_bz=eval_bz, rel_predict=0
            )

            per_snapshot_records.append({
                "snapshot_index": time_idx,
                "mrr_filter": mrr_filter_snap,
                "mrr_raw": mrr_snap,
                "num_triples": len(test_snap),
            })

            # roll history window forward (single-step, ground-truth history)
            input_list.pop(0)
            input_list.append(test_snap)

    results_df = pd.DataFrame(per_snapshot_records)

    # --- Correlation: snapshot index vs. error (1 - mrr_filter) ---
    results_df["error"] = 1.0 - results_df["mrr_filter"]
    rho, pval = stats.spearmanr(results_df["snapshot_index"], results_df["error"])

    corr_stats = {
        "spearman_rho": rho,
        "p_value": pval,
        "n_snapshots": len(results_df),
        "summary": (
            f"Spearman correlation between snapshot index and error (1 - MRR): "
            f"rho={rho:.3f}, p={pval:.4g}, n={len(results_df)} snapshots. "
            + (
                "Statistically significant positive correlation supports the "
                "'later snapshots are harder' assumption."
                if (pval < 0.05 and rho > 0)
                else "No statistically significant positive correlation was found; "
                     "the temporal-difficulty assumption is not strongly supported "
                     "for this dataset."
            )
        ),
    }

    print("\n" + "=" * 70)
    print(corr_stats["summary"])
    print("=" * 70 + "\n")

    # --- Save CSV ---
    csv_path = f"{save_prefix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved per-snapshot results to {csv_path}")

    # --- Plot ---
    _plot_temporal_difficulty(results_df, corr_stats, save_prefix)

    return results_df, corr_stats


def _plot_temporal_difficulty(results_df, corr_stats, save_prefix, figsize=(9, 6)):
    """
    Scatter of snapshot index vs. MRR, with a smoothed trend line and
    the Spearman correlation annotated on the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = results_df["snapshot_index"].values
    y = results_df["mrr_filter"].values

    ax.scatter(x, y, color="#4C72B0", alpha=0.6, s=30, label="Per-snapshot MRR", zorder=3)

    # smoothed trend line (quadratic fit, matches style used elsewhere in the paper's plots)
    if len(x) > 3:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_smooth, p(x_smooth), color="#C44E52", linewidth=2.5,
                 linestyle="--", label="Quadratic trend", zorder=2)

    ax.set_xlabel("Snapshot Index (Temporal Position)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Filtered MRR", fontsize=13, fontweight="bold")
    ax.set_title(
        "Per-Snapshot Model Performance vs. Temporal Position\n"
        f"Spearman ρ = {corr_stats['spearman_rho']:.3f}, p = {corr_stats['p_value']:.4g}",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = f"{save_prefix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


def analyze_entity_novelty(train_list, test_list, save_prefix="entity_novelty"):
    """
    Optional companion analysis: tracks the fraction of test-snapshot
    entities that never appeared in earlier training snapshots, per
    snapshot index. Rising novelty over time supports a churn-driven
    explanation for why later snapshots may be harder (rather than pure
    temporal recency), directly addressing the "more history = easier"
    counter-hypothesis.
    """
    seen_entities = set()
    novelty_records = []

    for t, snap in enumerate(train_list):
        entities_in_snap = set(snap[:, 0]).union(set(snap[:, 2]))
        seen_entities.update(entities_in_snap)

    # seen_entities now holds everything seen during training;
    # measure novelty against that fixed set across the test sequence
    running_seen = set(seen_entities)
    for t, snap in enumerate(test_list):
        entities_in_snap = set(snap[:, 0]).union(set(snap[:, 2]))
        novel = entities_in_snap - running_seen
        novelty_fraction = len(novel) / max(len(entities_in_snap), 1)
        novelty_records.append({
            "snapshot_index": t,
            "novelty_fraction": novelty_fraction,
            "num_entities": len(entities_in_snap),
            "num_novel": len(novel),
        })
        running_seen.update(entities_in_snap)

    df = pd.DataFrame(novelty_records)
    rho, pval = stats.spearmanr(df["snapshot_index"], df["novelty_fraction"])
    print(f"\nEntity novelty vs. snapshot index: rho={rho:.3f}, p={pval:.4g}")

    df.to_csv(f"{save_prefix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["snapshot_index"], df["novelty_fraction"], color="#55A868",
             linewidth=2, marker="o", markersize=3, alpha=0.8)
    ax.set_xlabel("Snapshot Index (Temporal Position)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of Novel Entities", fontsize=13, fontweight="bold")
    ax.set_title(
        "Entity Novelty Over Time\n"
        f"Spearman ρ = {rho:.3f}, p = {pval:.4g}",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved entity novelty plot to {save_prefix}.png")

    return df, {"spearman_rho": rho, "p_value": pval}


if __name__ == "__main__":
    print(
        "This module is meant to be imported into your existing training/eval "
        "pipeline. Example:\n\n"
        "    from temporal_difficulty_analysis import analyze_temporal_difficulty, analyze_entity_novelty\n\n"
        "    results_df, corr_stats = analyze_temporal_difficulty(\n"
        "        model, train_list, test_list, num_rels, num_nodes,\n"
        "        use_cuda, all_ans_list_test, static_graph, args,\n"
        "        save_prefix='ICEWS14_temporal_difficulty'\n"
        "    )\n\n"
        "    novelty_df, novelty_stats = analyze_entity_novelty(\n"
        "        train_list, test_list, save_prefix='ICEWS14_entity_novelty'\n"
        "    )\n"
    )