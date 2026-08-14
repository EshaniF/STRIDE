"""
temp_diff_analysis.py

Empirically tests the "later snapshots are harder" assumption (temporal_score =
t / (len(train_list)-1) in GeneralDifficultyAnalyzer) -- the single biggest
complaint from R1 and R2. Correlates snapshot temporal position against
per-snapshot filtered MRR under a *fixed, already-trained* model, so the
result isn't confounded by training order (a snapshot trained on late in an
epoch has seen more gradient steps than one trained on early -- evaluating
train-time loss would conflate "intrinsically hard" with "recently updated
on"). Evaluating a frozen model snapshot-by-snapshot on held-out data avoids
that confound entirely.

FIX vs. the previous version of this file: `analyze_temporal_difficulty` was
calling `model.predict(history_glist, num_rels, static_graph,
test_triples_input, use_cuda)` and unpacking 3 return values. That is the
RE-GCN/TIRGN-style signature, not this model's. The STRIDE/LogCL
`RecurrentRGCN.predict()` (see `run_pipeline.py`'s `test()`) requires:

    predict(que_pair, sub_graph, T_id, test_graph, num_rels, static_graph,
            test_triplets, use_cuda) -> (all_triples, scores_en)

i.e. two extra required arguments (`que_pair` from `e2r()`, `sub_graph` from
`get_sample_from_history_graph3()`, which itself needs a `sr_to_sro` dict
loaded from `../data/{dataset}/his_dict/train_s_r.npy`) and only 2 return
values, not 3. As written before, this call would raise a TypeError the
moment it ran -- the "empirical justification" was never actually being
computed. This version reproduces the exact call sequence `test()` uses,
including averaging raw + inverse-direction filtered MRR per snapshot, so the
correlation is measured against the *same* metric reported in the main
results table (not some other unreported quantity).

Usage (matches the call already added to run_pipeline.py's run_experiment):

    from temp_diff_analysis import analyze_temporal_difficulty, analyze_entity_novelty

    results_df, corr_stats = analyze_temporal_difficulty(
        model, train_list + valid_list, test_list, num_rels, num_nodes,
        use_cuda, all_ans_list_test, static_graph, args,
        save_prefix=f"{args.dataset}_temporal_difficulty"
    )
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from rgcn.utils import build_sub_graph, build_graph, get_total_rank
from rgcn import utils


# ---------------------------------------------------------------------------
# Local copies of the two helpers `model.predict()` depends on. These are
# duplicated (not imported) from run_pipeline.py to avoid a circular import
# (run_pipeline.py imports this module at the top). Keep in sync if you
# change either function there.
# ---------------------------------------------------------------------------

def _e2r(triplets, num_rels, use_cuda, gpu):
    src, rel, dst = triplets.transpose()
    uniq_e = np.unique(src)
    e_to_r = defaultdict(set)
    for j, (s, r, o) in enumerate(triplets):
        e_to_r[s].add(r)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx, idx + len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e_t = torch.from_numpy(np.array(uniq_e)).long()
    r_len_t = torch.from_numpy(np.array(r_len)).long()
    r_idx_t = torch.from_numpy(np.array(r_idx)).long()
    if use_cuda:
        uniq_e_t, r_len_t, r_idx_t = uniq_e_t.cuda(), r_len_t.cuda(), r_idx_t.cuda()
    return [uniq_e_t, r_len_t, r_idx_t]


def _get_sample_from_history_graph3(subg_arr, sr_to_sro, triples, num_nodes, num_rels, use_cuda, gpu):
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 2])

    er_list = list(set([(tri[0], tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0], tri[1]) for tri in inverse_triples]))

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})

    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic = pd.DataFrame({'sr': keys, 'dst': values})

    dst_df = df_dic.query('sr in @er_list')
    dst_get = dst_df['dst'].values
    two_ent = set().union(*dst_get) if len(dst_get) else set()
    all_ent = list(src_set | two_ent)
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')
    dst_get_inv = dst_df_inv['dst'].values
    two_ent_inv = set().union(*dst_get_inv) if len(dst_get_inv) else set()
    all_ent_inv = list(dst_set | two_ent_inv)
    result_inv = subg_df.query('src in @all_ent_inv')

    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu)
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return his_sub, his_sub_inv


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
    Evaluates the trained model snapshot-by-snapshot over `test_list` (same
    protocol/model-call sequence as run_pipeline.py's `test()` in "test" mode,
    minus the multi-step rollout), records per-snapshot filtered MRR
    (raw+inverse averaged, matching the reported metric), and correlates
    snapshot index against error.

    Returns:
        results_df: one row per test snapshot (snapshot_index, mrr_filter,
                     mrr_raw, num_triples, error)
        corr_stats: dict with spearman rho, p-value, and an explicit
                    supports/contradicts/inconclusive verdict for the
                    "later = harder" assumption.
    """
    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    # Built once, same as test(): the full concatenated history triple pool
    # used for second-order neighbor sampling in get_sample_from_history_graph3.
    subg_arr = np.concatenate(history_list[:])
    sr_to_sro = np.load(
        '../data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True
    ).item()

    per_snapshot_records = []

    with torch.no_grad():
        for time_idx, test_snap in enumerate(tqdm(test_list, desc="Evaluating per-snapshot MRR")):
            history_glist = [
                build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                for g in input_list
            ]

            inverse_triples = test_snap[:, [2, 1, 0]]
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels

            que_pair = _e2r(test_snap, num_rels, use_cuda, args.gpu)
            que_pair_inv = _e2r(inverse_triples, num_rels, use_cuda, args.gpu)

            sub_snap, sub_snap_inv = _get_sample_from_history_graph3(
                subg_arr, sr_to_sro, test_snap, num_nodes, num_rels, use_cuda, args.gpu
            )

            test_triples_input = (
                torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
            )
            test_triples_input_inv = (
                torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
            )

            # Matches model.predict()'s actual signature:
            # predict(que_pair, sub_graph, T_id, test_graph, num_rels,
            #         static_graph, test_triplets, use_cuda) -> (all_triples, scores_en)
            test_triples, final_score = model.predict(
                que_pair, sub_snap, time_idx, history_glist, num_rels,
                static_graph, test_triples_input, use_cuda
            )
            inv_test_triples, inv_final_score = model.predict(
                que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels,
                static_graph, test_triples_input_inv, use_cuda
            )

            mrr_filter_snap, mrr_snap, _, _ = get_total_rank(
                test_triples, final_score, all_ans_list[time_idx], eval_bz=eval_bz, rel_predict=0
            )
            mrr_filter_snap_inv, mrr_snap_inv, _, _ = get_total_rank(
                inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=eval_bz, rel_predict=0
            )

            # Average raw+inverse, same as test()'s reported all_mrr_filter/all_mrr_raw.
            avg_mrr_filter = (mrr_filter_snap + mrr_filter_snap_inv) / 2
            avg_mrr_raw = (mrr_snap + mrr_snap_inv) / 2

            per_snapshot_records.append({
                "snapshot_index": time_idx,
                "temporal_position": time_idx / max(len(test_list) - 1, 1),  # matches temporal_score's formula
                "mrr_filter": avg_mrr_filter,
                "mrr_raw": avg_mrr_raw,
                "num_triples": len(test_snap),
            })

            # roll history window forward (single-step, ground-truth history)
            input_list.pop(0)
            input_list.append(test_snap)

    results_df = pd.DataFrame(per_snapshot_records)

    # --- Correlation: snapshot index vs. error (1 - mrr_filter) ---
    results_df["error"] = 1.0 - results_df["mrr_filter"]
    rho, pval = stats.spearmanr(results_df["snapshot_index"], results_df["error"])

    if pval < 0.05 and rho > 0.2:
        verdict = "SUPPORTS"
        verdict_text = (
            "Statistically significant positive correlation: later snapshots have "
            "higher error under a fixed, trained model. The 'later = harder' "
            "assumption in temporal_score is empirically supported for this dataset."
        )
    elif pval < 0.05 and rho < -0.2:
        verdict = "CONTRADICTS"
        verdict_text = (
            "Statistically significant NEGATIVE correlation: later snapshots have "
            "LOWER error -- richer accumulated history appears to make later "
            "snapshots easier, not harder, for this dataset. temporal_weight "
            "should likely be reduced (or made dataset-adaptive) rather than "
            "treated as a universal +monotonic prior."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            "No statistically significant monotonic relationship between snapshot "
            "index and error was found. Temporal position alone is a weak "
            "difficulty proxy for this dataset; the composite score's other "
            "components (frequency/degree/size) likely carry more signal here."
        )

    corr_stats = {
        "spearman_rho": rho,
        "p_value": pval,
        "n_snapshots": len(results_df),
        "verdict": verdict,
        "summary": (
            f"Spearman correlation between snapshot index and error (1 - MRR): "
            f"rho={rho:.3f}, p={pval:.4g}, n={len(results_df)} snapshots.\n"
            f"[{verdict}] {verdict_text}"
        ),
    }

    print("\n" + "=" * 70)
    print(f"TEMPORAL DIFFICULTY CHECK -- {args.dataset}")
    print("=" * 70)
    print(corr_stats["summary"])
    print("=" * 70 + "\n")

    csv_path = f"{save_prefix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved per-snapshot results to {csv_path}")

    _plot_temporal_difficulty(results_df, corr_stats, save_prefix, dataset_name=args.dataset)

    return results_df, corr_stats


def _plot_temporal_difficulty(results_df, corr_stats, save_prefix, dataset_name="", figsize=(9, 6)):
    """
    Scatter of snapshot index vs. MRR, with a smoothed trend line and the
    Spearman correlation + verdict annotated on the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = results_df["snapshot_index"].values
    y = results_df["mrr_filter"].values

    ax.scatter(x, y, color="#4C72B0", alpha=0.6, s=30, label="Per-snapshot filtered MRR", zorder=3)

    if len(x) > 3:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_smooth, p(x_smooth), color="#C44E52", linewidth=2.5,
                 linestyle="--", label="Quadratic trend", zorder=2)

    ax.set_xlabel("Snapshot Index (Temporal Position)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Filtered MRR", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Per-Snapshot Model Performance vs. Temporal Position -- {dataset_name}\n"
        f"Spearman ρ = {corr_stats['spearman_rho']:.3f}, p = {corr_stats['p_value']:.4g}  "
        f"[{corr_stats['verdict']}]",
        fontsize=12, fontweight="bold"
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
    Companion analysis (unchanged -- this one had no signature bug): tracks
    the fraction of test-snapshot entities never seen in training, per
    snapshot index. Rising novelty over time supports a churn-driven
    explanation for later-snapshot difficulty (rather than pure temporal
    recency), directly addressing the "more history = easier" counter-
    hypothesis R1/R2 raised.
    """
    seen_entities = set()
    for t, snap in enumerate(train_list):
        entities_in_snap = set(snap[:, 0]).union(set(snap[:, 2]))
        seen_entities.update(entities_in_snap)

    running_seen = set(seen_entities)
    novelty_records = []
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
        "This module is meant to be imported into run_pipeline.py. Example:\n\n"
        "    from temp_diff_analysis import analyze_temporal_difficulty, analyze_entity_novelty\n\n"
        "    results_df, corr_stats = analyze_temporal_difficulty(\n"
        "        model, train_list + valid_list, test_list, num_rels, num_nodes,\n"
        "        use_cuda, all_ans_list_test, static_graph, args,\n"
        "        save_prefix=f'{args.dataset}_temporal_difficulty'\n"
        "    )\n\n"
        "    novelty_df, novelty_stats = analyze_entity_novelty(\n"
        "        train_list, test_list, save_prefix=f'{args.dataset}_entity_novelty'\n"
        "    )\n"
    )