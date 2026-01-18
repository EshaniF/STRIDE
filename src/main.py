import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from diagnose import diagnose_difficulty_computation


###################################
def compute_test_difficulty_scores(train_list, test_list, num_rels):
    """
    Compute difficulty scores specifically for TEST snapshots
    based on what the model sees at test time
    """
    print("\n" + "="*80)
    print("Computing Difficulty Scores for TEST Set")
    print("="*80)
    
    from collections import defaultdict
    
    # First, gather statistics from ALL available data (train)
    all_train_triples = np.concatenate(train_list) if train_list else np.array([])
    
    entity_freq = defaultdict(int)
    relation_freq = defaultdict(int)
    sr_pairs = defaultdict(set)
    entity_relations = defaultdict(set)
    
    for triple in all_train_triples:
        s, r, o = triple
        entity_freq[s] += 1
        entity_freq[o] += 1
        relation_freq[r] += 1
        sr_pairs[(s, r)].add(o)
        entity_relations[s].add(r)
        entity_relations[o].add(r)
    
    # Now compute difficulty for each TEST snapshot
    test_difficulty_scores = []
    
    print(f"\nAnalyzing {len(test_list)} test snapshots...")
    
    for t, test_snap in enumerate(test_list):
        if len(test_snap) == 0:
            test_difficulty_scores.append(0.5)
            continue
        
        # === Component 1: Answer Ambiguity ===
        # How many possible answers exist for queries in this test snapshot?
        ambiguity_scores = []
        for triple in test_snap:
            s, r, o = triple
            num_possible = len(sr_pairs.get((s, r), set()))
            ambiguity = min(num_possible / 10.0, 1.0)
            ambiguity_scores.append(ambiguity)
        
        ambiguity_score = np.mean(ambiguity_scores) if ambiguity_scores else 0.5
        
        # === Component 2: Entity Rarity ===
        # Are the entities in this snapshot rare (seen infrequently in training)?
        rarity_scores = []
        for triple in test_snap:
            s, r, o = triple
            s_freq = entity_freq.get(s, 0)  # 0 if never seen in training!
            o_freq = entity_freq.get(o, 0)
            
            # Unseen entities = maximum difficulty
            if s_freq == 0 or o_freq == 0:
                rarity = 1.0
            else:
                max_freq = max(entity_freq.values())
                s_rarity = 1.0 - (s_freq / max_freq)
                o_rarity = 1.0 - (o_freq / max_freq)
                rarity = (s_rarity + o_rarity) / 2.0
            
            rarity_scores.append(rarity)
        
        rarity_score = np.mean(rarity_scores) if rarity_scores else 0.5
        
        # === Component 3: Relation Selectivity ===
        selectivity_scores = []
        for triple in test_snap:
            s, r, o = triple
            r_freq = relation_freq.get(r, 0)
            
            if r_freq == 0:
                selectivity = 1.0
            else:
                max_rel_freq = max(relation_freq.values())
                selectivity = r_freq / max_rel_freq
            
            selectivity_scores.append(selectivity)
        
        selectivity_score = np.mean(selectivity_scores) if selectivity_scores else 0.5
        
        # === Component 4: Novel Entities ===
        # What percentage of entities in test were NEVER seen in training?
        test_entities = set()
        for triple in test_snap:
            test_entities.add(triple[0])
            test_entities.add(triple[2])
        
        train_entities = set(entity_freq.keys())
        novel_entities = test_entities - train_entities
        novelty_score = len(novel_entities) / len(test_entities) if test_entities else 0.0
        
        # === Component 5: Temporal Distance ===
        # Later test snapshots are potentially harder (further from training)
        temporal_score = t / max(len(test_list) - 1, 1)
        
        # === Combine Components ===
        combined_score = (
            0.30 * ambiguity_score +      # Answer ambiguity
            0.25 * rarity_score +          # Entity rarity
            0.15 * selectivity_score +     # Relation selectivity
            0.20 * novelty_score +         # Novel entities
            0.10 * temporal_score          # Temporal progression
        )
        
        test_difficulty_scores.append(combined_score)
        
        # Print details for first few snapshots
        if t < 3:
            print(f"\nTest Snapshot {t}:")
            print(f"  Triples: {len(test_snap)}")
            print(f"  Novel entities: {len(novel_entities)}/{len(test_entities)} ({novelty_score:.1%})")
            print(f"  Ambiguity: {ambiguity_score:.3f}")
            print(f"  Rarity: {rarity_score:.3f}")
            print(f"  Combined difficulty: {combined_score:.3f}")
    
    test_difficulty_scores = np.array(test_difficulty_scores)
    
    # Normalize to [0, 1] range
    score_min = test_difficulty_scores.min()
    score_max = test_difficulty_scores.max()
    score_range = score_max - score_min
    
    if score_range > 1e-6:
        test_difficulty_scores = (test_difficulty_scores - score_min) / score_range
    
    print(f"\nTest Difficulty Statistics:")
    print(f"  Range: [{test_difficulty_scores.min():.3f}, {test_difficulty_scores.max():.3f}]")
    print(f"  Mean: {test_difficulty_scores.mean():.3f}")
    print(f"  Std: {test_difficulty_scores.std():.3f}")
    
    return test_difficulty_scores


def analyze_test_difficulty_vs_performance(test_list, test_difficulty_scores, 
                                          test_mrr, test_h1, test_h10):
    """
    Properly analyze difficulty vs performance for TEST set
    """
    print("\n" + "="*80)
    print("DIFFICULTY VS PERFORMANCE ANALYSIS (TEST SET)")
    print("="*80)
    
    # Verify alignment
    n_test = len(test_list)
    n_diff = len(test_difficulty_scores)
    n_perf = len(test_mrr)
    
    print(f"\nData Alignment Check:")
    print(f"  Test snapshots: {n_test}")
    print(f"  Difficulty scores: {n_diff}")
    print(f"  Performance metrics: {n_perf}")
    
    if not (n_test == n_diff == n_perf):
        print(f"\n⚠️  WARNING: Misalignment detected!")
        min_len = min(n_test, n_diff, n_perf)
        print(f"  Using first {min_len} snapshots for analysis")
        test_difficulty_scores = test_difficulty_scores[:min_len]
        test_mrr = test_mrr[:min_len]
        test_h1 = test_h1[:min_len]
        test_h10 = test_h10[:min_len]
    
    # Calculate correlations
    corr_mrr, p_mrr = pearsonr(test_difficulty_scores, test_mrr)
    corr_h1, p_h1 = pearsonr(test_difficulty_scores, test_h1)
    corr_h10, p_h10 = pearsonr(test_difficulty_scores, test_h10)
    
    print(f"\nCorrelation Analysis:")
    print(f"  Difficulty vs MRR:     {corr_mrr:+.3f} (p={p_mrr:.4f})")
    print(f"  Difficulty vs Hits@1:  {corr_h1:+.3f} (p={p_h1:.4f})")
    print(f"  Difficulty vs Hits@10: {corr_h10:+.3f} (p={p_h10:.4f})")
    
    if abs(corr_mrr) > 0.3:
        print(f"\n✓ Good correlation! Difficulty scores are meaningful.")
    elif abs(corr_mrr) > 0.15:
        print(f"\n~ Moderate correlation. Difficulty captures some signal.")
    else:
        print(f"\n✗ Weak correlation. Difficulty may not be well-calibrated.")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Scatter plots
    axes[0, 0].scatter(test_difficulty_scores, test_mrr, alpha=0.6, s=80)
    axes[0, 0].set_xlabel('Difficulty Score')
    axes[0, 0].set_ylabel('MRR')
    axes[0, 0].set_title(f'Difficulty vs MRR (corr={corr_mrr:+.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(test_difficulty_scores, test_mrr, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(test_difficulty_scores, p(test_difficulty_scores), "r--", alpha=0.8, linewidth=2)
    
    axes[0, 1].scatter(test_difficulty_scores, test_h1, alpha=0.6, s=80, c='orange')
    axes[0, 1].set_xlabel('Difficulty Score')
    axes[0, 1].set_ylabel('Hits@1')
    axes[0, 1].set_title(f'Difficulty vs Hits@1 (corr={corr_h1:+.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(test_difficulty_scores, test_h10, alpha=0.6, s=80, c='green')
    axes[0, 2].set_xlabel('Difficulty Score')
    axes[0, 2].set_ylabel('Hits@10')
    axes[0, 2].set_title(f'Difficulty vs Hits@10 (corr={corr_h10:+.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Time series
    snapshot_indices = np.arange(len(test_difficulty_scores))
    
    ax_diff = axes[1, 0]
    ax_diff.plot(snapshot_indices, test_difficulty_scores, 'b-o', linewidth=2, markersize=6)
    ax_diff.set_xlabel('Test Snapshot Index')
    ax_diff.set_ylabel('Difficulty Score')
    ax_diff.set_title('Difficulty Over Test Snapshots')
    ax_diff.grid(True, alpha=0.3)
    
    ax_mrr = axes[1, 1]
    ax_mrr.plot(snapshot_indices, test_mrr, 'r-o', linewidth=2, markersize=6)
    ax_mrr.set_xlabel('Test Snapshot Index')
    ax_mrr.set_ylabel('MRR')
    ax_mrr.set_title('MRR Over Test Snapshots')
    ax_mrr.grid(True, alpha=0.3)
    
    ax_both = axes[1, 2]
    ax1 = ax_both
    ax1.plot(snapshot_indices, test_difficulty_scores, 'b-o', linewidth=2, markersize=6, label='Difficulty')
    ax1.set_xlabel('Test Snapshot Index')
    ax1.set_ylabel('Difficulty', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(snapshot_indices, test_mrr, 'r-s', linewidth=2, markersize=6, label='MRR')
    ax2.set_ylabel('MRR', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Difficulty and MRR Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig('test_difficulty_vs_performance.png', dpi=150, bbox_inches='tight')
    
    print(f"\n{'='*80}")
    print(f"Analysis saved to 'test_difficulty_vs_performance.png'")
    print(f"{'='*80}")
    
    return corr_mrr, corr_h1, corr_h10
# class PerformanceTracker:
#     """
#     Tracks performance metrics per snapshot for difficulty analysis
#     """
#     def __init__(self, num_snapshots):
#         self.num_snapshots = num_snapshots
#         self.snapshot_metrics = defaultdict(lambda: {
#             'mrr': [],
#             'hits@1': [],
#             'hits@3': [],
#             'hits@10': [],
#             'count': 0
#         })
        
#     def update(self, snapshot_idx, mrr, hits_at_k):
#         """Update metrics for a specific snapshot"""
#         self.snapshot_metrics[snapshot_idx]['mrr'].append(mrr)
#         self.snapshot_metrics[snapshot_idx]['hits@1'].append(hits_at_k[0])
#         self.snapshot_metrics[snapshot_idx]['hits@3'].append(hits_at_k[1])
#         self.snapshot_metrics[snapshot_idx]['hits@10'].append(hits_at_k[2])
#         self.snapshot_metrics[snapshot_idx]['count'] += 1
    
#     def get_average_metrics(self):
#         """Get average metrics per snapshot"""
#         avg_metrics = {}
#         for idx in range(self.num_snapshots):
#             if idx in self.snapshot_metrics and self.snapshot_metrics[idx]['count'] > 0:
#                 avg_metrics[idx] = {
#                     'mrr': np.mean(self.snapshot_metrics[idx]['mrr']),
#                     'hits@1': np.mean(self.snapshot_metrics[idx]['hits@1']),
#                     'hits@3': np.mean(self.snapshot_metrics[idx]['hits@3']),
#                     'hits@10': np.mean(self.snapshot_metrics[idx]['hits@10'])
#                 }
#             else:
#                 avg_metrics[idx] = {
#                     'mrr': 0.0,
#                     'hits@1': 0.0,
#                     'hits@3': 0.0,
#                     'hits@10': 0.0
#                 }
#         return avg_metrics


# def plot_difficulty_vs_performance(difficulty_scores, performance_metrics, dataset_name, save_path='analysis_plots'):
#     """
#     Create comprehensive difficulty vs performance visualizations
    
#     Args:
#         difficulty_scores: Array of difficulty scores per snapshot
#         performance_metrics: Dict with snapshot_idx -> metrics
#         dataset_name: Name of dataset for plot title
#         save_path: Directory to save plots
#     """
#     os.makedirs(save_path, exist_ok=True)
    
#     # Extract data
#     snapshots = sorted(performance_metrics.keys())
#     difficulties = [difficulty_scores[i] for i in snapshots]
#     mrrs = [performance_metrics[i]['mrr'] for i in snapshots]
#     hits1 = [performance_metrics[i]['hits@1'] for i in snapshots]
#     hits3 = [performance_metrics[i]['hits@3'] for i in snapshots]
#     hits10 = [performance_metrics[i]['hits@10'] for i in snapshots]
    
#     # Create figure with multiple subplots
#     fig = plt.figure(figsize=(20, 12))
    
#     # 1. Main scatter plot: Difficulty vs MRR
#     ax1 = plt.subplot(2, 3, 1)
#     scatter1 = ax1.scatter(difficulties, mrrs, c=snapshots, cmap='viridis', 
#                           s=100, alpha=0.6, edgecolors='black')
#     ax1.set_xlabel('Difficulty Score', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('MRR', fontsize=12, fontweight='bold')
#     ax1.set_title(f'Difficulty vs MRR - {dataset_name}', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Add trend line
#     z = np.polyfit(difficulties, mrrs, 2)
#     p = np.poly1d(z)
#     x_trend = np.linspace(min(difficulties), max(difficulties), 100)
#     ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend (2nd order)')
#     ax1.legend()
    
#     # Add correlation coefficient
#     corr = np.corrcoef(difficulties, mrrs)[0, 1]
#     ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
#              transform=ax1.transAxes, fontsize=11, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.colorbar(scatter1, ax=ax1, label='Snapshot Index')
    
#     # 2. Difficulty vs Hits@1
#     ax2 = plt.subplot(2, 3, 2)
#     scatter2 = ax2.scatter(difficulties, hits1, c=snapshots, cmap='plasma', 
#                           s=100, alpha=0.6, edgecolors='black')
#     ax2.set_xlabel('Difficulty Score', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('Hits@1', fontsize=12, fontweight='bold')
#     ax2.set_title(f'Difficulty vs Hits@1 - {dataset_name}', fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
    
#     # Add trend line
#     z = np.polyfit(difficulties, hits1, 2)
#     p = np.poly1d(z)
#     ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
#     ax2.legend()
    
#     corr_h1 = np.corrcoef(difficulties, hits1)[0, 1]
#     ax2.text(0.05, 0.95, f'Correlation: {corr_h1:.3f}', 
#              transform=ax2.transAxes, fontsize=11, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.colorbar(scatter2, ax=ax2, label='Snapshot Index')
    
#     # 3. Difficulty vs Hits@10
#     ax3 = plt.subplot(2, 3, 3)
#     scatter3 = ax3.scatter(difficulties, hits10, c=snapshots, cmap='coolwarm', 
#                           s=100, alpha=0.6, edgecolors='black')
#     ax3.set_xlabel('Difficulty Score', fontsize=12, fontweight='bold')
#     ax3.set_ylabel('Hits@10', fontsize=12, fontweight='bold')
#     ax3.set_title(f'Difficulty vs Hits@10 - {dataset_name}', fontsize=14, fontweight='bold')
#     ax3.grid(True, alpha=0.3)
    
#     # Add trend line
#     z = np.polyfit(difficulties, hits10, 2)
#     p = np.poly1d(z)
#     ax3.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
#     ax3.legend()
    
#     corr_h10 = np.corrcoef(difficulties, hits10)[0, 1]
#     ax3.text(0.05, 0.95, f'Correlation: {corr_h10:.3f}', 
#              transform=ax3.transAxes, fontsize=11, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.colorbar(scatter3, ax=ax3, label='Snapshot Index')
    
#     # 4. Binned difficulty analysis
#     ax4 = plt.subplot(2, 3, 4)
    
#     # Create difficulty bins
#     n_bins = 5
#     bin_edges = np.linspace(min(difficulties), max(difficulties), n_bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     binned_mrr = []
#     binned_mrr_std = []
#     for i in range(n_bins):
#         mask = (np.array(difficulties) >= bin_edges[i]) & (np.array(difficulties) < bin_edges[i+1])
#         if np.sum(mask) > 0:
#             binned_mrr.append(np.mean(np.array(mrrs)[mask]))
#             binned_mrr_std.append(np.std(np.array(mrrs)[mask]))
#         else:
#             binned_mrr.append(0)
#             binned_mrr_std.append(0)
    
#     ax4.bar(bin_centers, binned_mrr, width=(bin_edges[1]-bin_edges[0])*0.8, 
#             alpha=0.7, color='steelblue', edgecolor='black', yerr=binned_mrr_std, capsize=5)
#     ax4.set_xlabel('Difficulty Score (Binned)', fontsize=12, fontweight='bold')
#     ax4.set_ylabel('Average MRR', fontsize=12, fontweight='bold')
#     ax4.set_title('Performance by Difficulty Bins', fontsize=14, fontweight='bold')
#     ax4.grid(True, alpha=0.3, axis='y')
    
#     # Add sample counts
#     for i, center in enumerate(bin_centers):
#         mask = (np.array(difficulties) >= bin_edges[i]) & (np.array(difficulties) < bin_edges[i+1])
#         count = np.sum(mask)
#         ax4.text(center, binned_mrr[i] + binned_mrr_std[i] + 0.01, f'n={count}', 
#                 ha='center', fontsize=9)
    
#     # 5. Temporal progression with difficulty overlay
#     ax5 = plt.subplot(2, 3, 5)
    
#     ax5_twin = ax5.twinx()
#     line1 = ax5.plot(snapshots, mrrs, 'b-o', linewidth=2, markersize=6, label='MRR')
#     line2 = ax5_twin.plot(snapshots, difficulties, 'r--s', linewidth=2, markersize=6, label='Difficulty')
    
#     ax5.set_xlabel('Snapshot Index', fontsize=12, fontweight='bold')
#     ax5.set_ylabel('MRR', fontsize=12, fontweight='bold', color='b')
#     ax5_twin.set_ylabel('Difficulty Score', fontsize=12, fontweight='bold', color='r')
#     ax5.set_title('Temporal Evolution: MRR vs Difficulty', fontsize=14, fontweight='bold')
#     ax5.tick_params(axis='y', labelcolor='b')
#     ax5_twin.tick_params(axis='y', labelcolor='r')
#     ax5.grid(True, alpha=0.3)
    
#     # Combined legend
#     lines = line1 + line2
#     labels = [l.get_label() for l in lines]
#     ax5.legend(lines, labels, loc='best')
    
#     # 6. All metrics comparison
#     ax6 = plt.subplot(2, 3, 6)
    
#     # Normalize difficulties for comparison
#     norm_diff = (np.array(difficulties) - min(difficulties)) / (max(difficulties) - min(difficulties))
    
#     ax6.plot(snapshots, mrrs, 'b-o', label='MRR', linewidth=2, markersize=5)
#     ax6.plot(snapshots, hits1, 'g-s', label='Hits@1', linewidth=2, markersize=5)
#     ax6.plot(snapshots, hits10, 'orange', marker='^', label='Hits@10', linewidth=2, markersize=5)
#     ax6.plot(snapshots, norm_diff, 'r--', label='Difficulty (norm)', linewidth=2, alpha=0.7)
    
#     ax6.set_xlabel('Snapshot Index', fontsize=12, fontweight='bold')
#     ax6.set_ylabel('Score', fontsize=12, fontweight='bold')
#     ax6.set_title('All Metrics vs Difficulty Over Time', fontsize=14, fontweight='bold')
#     ax6.legend(loc='best')
#     ax6.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f'{save_path}/difficulty_vs_performance_{dataset_name}.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Save detailed statistics to CSV
#     stats_data = []
#     for i, snapshot in enumerate(snapshots):
#         stats_data.append({
#             'snapshot': snapshot,
#             'difficulty': difficulties[i],
#             'mrr': mrrs[i],
#             'hits@1': hits1[i],
#             'hits@3': hits3[i],
#             'hits@10': hits10[i]
#         })
    
#     stats_filename = f'{save_path}/difficulty_performance_stats_{dataset_name}.csv'
#     with open(stats_filename, 'w', newline='') as f:
#         fieldnames = ['snapshot', 'difficulty', 'mrr', 'hits@1', 'hits@3', 'hits@10']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(stats_data)
    
#     print(f"\nDifficulty vs Performance Analysis:")
#     print(f"  - Correlation (Difficulty vs MRR): {corr:.3f}")
#     print(f"  - Correlation (Difficulty vs Hits@1): {corr_h1:.3f}")
#     print(f"  - Correlation (Difficulty vs Hits@10): {corr_h10:.3f}")
#     print(f"  - Plots saved to: {save_path}/difficulty_vs_performance_{dataset_name}.png")
#     print(f"  - Statistics saved to: {stats_filename}")
    
#     return {
#         'correlation_mrr': corr,
#         'correlation_hits1': corr_h1,
#         'correlation_hits10': corr_h10,
#         'binned_performance': list(zip(bin_centers, binned_mrr, binned_mrr_std))
#     }
# ####################################
def update_dict(subg_arr, s_to_sro, sr_to_sro,sro_to_fre, num_rels):
    # Update the query based on the input graph at each time 
    inverse_subg = subg_arr[:, [2, 1, 0]]
    #creating new relations for inverse of existing ones
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    #join both triplets
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        #s_sro gives a dictionary sourceentity : sro {1: {(1, 40, 0), (1, 10, 0)}, 2: {(2, 30, 1)}, 4: {(4, 30, 5)}}
        s_to_sro[src].add((src, rel, dst))
        #sr_sro gives a dictionary (sourc, rel) : dst {(1, 10): {0}, (2, 20): {1}, (4, 30): {5}, (1, 40): {0}, (2, 30): {1}})
        sr_to_sro[(src, rel)].add(dst)
        
def e2r(triplets, num_rels):
    # Statistics on the same query entity connecting different relationships
    src, rel, dst = triplets.transpose()
    # get all relations
    # uniq_e = np.concatenate((src, dst))
    uniq_e = np.unique(src)
    # generate r2e
    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)
        # e_to_r[dst].add(rel+num_rels)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    #uniq_e returns unique subject entities in triplets
    #r_idx reurns relation idxs
    return [uniq_e, r_len, r_idx]

def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,num_nodes, num_rels, use_cuda, gpu):
    # q_to_sro = defaultdict(list)
    q_to_sro = set()
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])
    # ent_set = set(all_triples[:, 0])
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])

    # ----------------二阶邻居采样 Second-order neighbor sampling-----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    #整合重复三元组并统计三元组的频率，将三元组的频率作为第四列数据
    # Integrate repeated triples and count the frequency of triples, using the frequency of triples as the fourth column of data
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #将查询字段转化为pandas - Convert query field to pandas

    dst_df = df_dic.query('sr in @er_list')  #获取查询实体和关系的pandas - Get query entities and relationships pandas
    dst_get = dst_df['dst'].values    #获取目标尾实体 - Get the target tail entity
    two_ent = set().union(*dst_get)   #将头实体与尾实体进行整合 - Integrate head and tail entities
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  #获取查询实体和关系的pandas - Get query entities and relationships pandas
    dst_get_inv = dst_df_inv['dst'].values    #获取目标尾实体 - Get the target tail entity
    two_ent_inv = set().union(*dst_get_inv)   #将头实体与尾实体进行整合 - Integrate head and tail entities
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')
    #----------------二阶邻居采样 Second-order neighbor sampling-----------------------
    # result = subg_df.query('src in @src_set')
    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu) 
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv

class GeneralCurriculumScheduler:
    """
    General curriculum learning scheduler that adapts to any temporal knowledge graph dataset.
    Uses adaptive strategies based on data characteristics discovered during training.
    """
    
    def __init__(self, total_epochs=50, warmup_ratio=0.4, strategy='adaptive'):
        """
        Args:
            total_epochs: Total number of training epochs
            warmup_ratio: Fraction of epochs for curriculum warmup (0.4 = 40%)
            strategy: 'adaptive', 'linear', or 'cosine'
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.strategy = strategy
        self.start_ratio = 0.2  # Start with 20% of data
        
    def get_current_ratio(self, epoch):
        """Calculate current curriculum ratio based on epoch and strategy"""
        if epoch >= self.warmup_epochs:
            return 1.0
            
        progress = epoch / self.warmup_epochs
        
        if self.strategy == 'adaptive':
            # Adaptive strategy: starts slow, accelerates in middle, slows at end
            # Good for both highly skewed (GDELT) and regular (ICEWS) datasets
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (
                0.5 * (1 - np.cos(progress * np.pi)) + 0.3 * progress
            )
        elif self.strategy == 'cosine':
            # Smooth cosine progression
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (1 - np.cos(progress * np.pi)) / 2
        else:
            # Linear progression (default)
            ratio = self.start_ratio + (1.0 - self.start_ratio) * progress
            
        return min(ratio, 1.0)

class GeneralDifficultyAnalyzer:
    """
    General difficulty analyzer that works for any temporal knowledge graph dataset.
    Combines multiple difficulty metrics to handle various data characteristics.
    """
    
    def __init__(self, train_list, ablation_mode='all'):
        """
        Args:
            train_list: List of training snapshots
            ablation_mode: Which components to include
                - 'all': All 4 components (default)
                - 'first_3': Temporal, frequency, and degree (exclude size)
                - 'first_2': Temporal and frequency only (exclude degree and size)
                - 'first_1': Temporal only (exclude frequency, degree, and size)
                - 'none': Uniform difficulty (baseline)
        """
        self.train_list = train_list
        self.all_triples = np.concatenate(train_list) if train_list else np.array([])
        self.ablation_mode = ablation_mode
        self._compute_statistics()
        
    def _compute_statistics(self):
        """Compute comprehensive statistics for difficulty assessment"""
        if len(self.all_triples) == 0:
            self.entity_freq = defaultdict(int)
            self.relation_freq = defaultdict(int)
            self.entity_degree = defaultdict(set)
            return
            
        # Entity and relation frequencies
        self.entity_freq = defaultdict(int)
        self.relation_freq = defaultdict(int)
        self.entity_degree = defaultdict(set)
        
        for triple in self.all_triples:
            s, r, o = triple
            self.entity_freq[s] += 1
            self.entity_freq[o] += 1
            self.relation_freq[r] += 1
            # Track entity degrees (number of unique relations)
            self.entity_degree[s].add(r)
            self.entity_degree[o].add(r)
        
        # Compute distribution statistics for adaptive difficulty
        entity_freqs = list(self.entity_freq.values())
        self.entity_freq_std = np.std(entity_freqs) if entity_freqs else 0
        self.entity_freq_mean = np.mean(entity_freqs) if entity_freqs else 0
        
        relation_freqs = list(self.relation_freq.values())
        self.relation_freq_std = np.std(relation_freqs) if relation_freqs else 0
        self.relation_freq_mean = np.mean(relation_freqs) if relation_freqs else 0
        
    def compute_difficulty_scores(self):
        """
        Compute difficulty scores with enhanced variance and proper scaling.
        Uses multiple strategies to ensure meaningful score distribution.
        """
        difficulty_scores = []
        
        # Strategy 1: Collect raw metrics for all snapshots first
        snapshot_metrics = []
        
        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                snapshot_metrics.append({
                    'temporal': 0,
                    'frequency': 0,
                    'degree': 0,
                    'size': 0,
                    'entities': set(),
                    'relations': set()
                })
                continue
            
            # Temporal progression (0 to 1)
            temporal_raw = t / max(len(self.train_list) - 1, 1)
            
            # Entity/relation rarity (inverse frequency with log scaling)
            entity_rarities = []
            relation_rarities = []
            entities_in_snap = set()
            relations_in_snap = set()
            
            for triple in snap:
                s, r, o = triple
                entities_in_snap.add(s)
                entities_in_snap.add(o)
                relations_in_snap.add(r)
                
                # Higher frequency = lower rarity score
                s_freq = max(self.entity_freq.get(s, 1), 1)
                o_freq = max(self.entity_freq.get(o, 1), 1)
                r_freq = max(self.relation_freq.get(r, 1), 1)
                
                # Inverse frequency with log dampening
                total_entities = len(self.entity_freq)
                total_relations = len(self.relation_freq)
                
                entity_rarity = (1.0 / s_freq + 1.0 / o_freq) / 2.0
                relation_rarity = 1.0 / r_freq
                
                entity_rarities.append(entity_rarity)
                relation_rarities.append(relation_rarity)
            
            # Structural complexity (average entity degree)
            degrees = []
            for triple in snap:
                s, r, o = triple
                s_degree = len(self.entity_degree.get(s, set()))
                o_degree = len(self.entity_degree.get(o, set()))
                degrees.append((s_degree + o_degree) / 2.0)
            
            snapshot_metrics.append({
                'temporal': temporal_raw,
                'frequency': np.mean(entity_rarities) * 0.7 + np.mean(relation_rarities) * 0.3,
                'degree': np.mean(degrees),
                'size': len(snap),
                'entities': entities_in_snap,
                'relations': relations_in_snap,
                'entity_diversity': len(entities_in_snap) / max(len(snap), 1),
                'relation_diversity': len(relations_in_snap) / max(len(snap), 1)
            })
        
        # Strategy 2: Normalize each metric using percentile-based approach
        # This ensures better distribution than min-max
        
        temporal_scores = [m['temporal'] for m in snapshot_metrics]
        frequency_scores = [m['frequency'] for m in snapshot_metrics]
        degree_scores = [m['degree'] for m in snapshot_metrics]
        size_scores = [m['size'] for m in snapshot_metrics]
        
        def percentile_normalize(scores):
            """Normalize using rank-based percentiles for better distribution"""
            if len(scores) <= 1:
                return [0.5] * len(scores)
            
            # Convert to ranks
            sorted_indices = np.argsort(scores)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(scores))
            
            # Normalize ranks to [0, 1]
            normalized = ranks / max(len(scores) - 1, 1)
            return normalized
        
        # Apply percentile normalization
        temporal_norm = np.array(temporal_scores)  # Already normalized
        frequency_norm = percentile_normalize(frequency_scores)
        degree_norm = percentile_normalize(degree_scores)
        size_norm = percentile_normalize(size_scores)
        
        # Strategy 3: Add diversity-based difficulty
        diversity_scores = []
        for m in snapshot_metrics:
            # Snapshots with more diverse entities/relations are harder
            div_score = (m['entity_diversity'] + m['relation_diversity']) / 2.0
            diversity_scores.append(div_score)
        diversity_norm = percentile_normalize(diversity_scores)
        
        # Strategy 4: Combine with amplified weights based on ablation mode
        for i in range(len(self.train_list)):
            if len(self.train_list[i]) == 0:
                difficulty_scores.append(0.0)
                continue
            
            if self.ablation_mode == 'none':
                difficulty_scores.append(0.5)
                continue
            
            if self.ablation_mode == 'first_1':
                # Only temporal
                score = temporal_norm[i]
                
            elif self.ablation_mode == 'first_2':
                # Temporal + Frequency with emphasis on variance
                score = (
                    0.5 * temporal_norm[i] +
                    0.5 * frequency_norm[i]
                )
                
            elif self.ablation_mode == 'first_3':
                # Temporal + Frequency + Degree
                score = (
                    0.4 * temporal_norm[i] +
                    0.35 * frequency_norm[i] +
                    0.25 * degree_norm[i]
                )
                
            else:  # 'all'
                # All components with balanced weights
                score = (
                    0.3 * temporal_norm[i] +
                    0.3 * frequency_norm[i] +
                    0.2 * degree_norm[i] +
                    0.1 * size_norm[i] +
                    0.1 * diversity_norm[i]
                )
            
            difficulty_scores.append(score)
        
        difficulty_scores = np.array(difficulty_scores)
        
        # Strategy 5: Apply non-linear transformation to enhance variance
        # Use power transformation to spread out middle values
        mean_score = difficulty_scores.mean()
        
        if mean_score > 0:
            # Apply a gentle power transformation
            # Values < 0.5 get pushed down, values > 0.5 get pushed up
            power = 1.5  # Adjust this to control spread (higher = more spread)
            
            # Center around 0.5, apply power, then scale back
            centered = difficulty_scores - 0.5
            sign = np.sign(centered)
            magnitude = np.abs(centered)
            transformed = sign * (magnitude ** power)
            difficulty_scores = transformed + 0.5
            
            # Ensure bounds [0, 1]
            difficulty_scores = np.clip(difficulty_scores, 0, 1)
        
        # Strategy 6: Final min-max normalization to use full range
        score_min = difficulty_scores.min()
        score_max = difficulty_scores.max()
        score_range = score_max - score_min
        
        if score_range > 1e-6:
            difficulty_scores = (difficulty_scores - score_min) / score_range
        else:
            # If all scores are the same, use temporal order
            difficulty_scores = temporal_norm
        
        # Diagnostic output
        print(f"\nDifficulty Score Statistics:")
        print(f"  Range: [{difficulty_scores.min():.4f}, {difficulty_scores.max():.4f}]")
        print(f"  Mean: {difficulty_scores.mean():.4f}, Std: {difficulty_scores.std():.4f}")
        print(f"  Quartiles: Q1={np.percentile(difficulty_scores, 25):.4f}, "
            f"Q2={np.percentile(difficulty_scores, 50):.4f}, "
            f"Q3={np.percentile(difficulty_scores, 75):.4f}")
        print(f"  First 5: {difficulty_scores[:5]}")
        print(f"  Last 5: {difficulty_scores[-5:]}")
        
        return np.array(difficulty_scores)

# class GeneralDifficultyAnalyzer:
#     """
#     General difficulty analyzer that works for any temporal knowledge graph dataset.
#     Combines multiple difficulty metrics to handle various data characteristics.
#     """
    
#     def __init__(self, train_list):
#         self.train_list = train_list
#         self.all_triples = np.concatenate(train_list) if train_list else np.array([])
#         self._compute_statistics()
        
#     def _compute_statistics(self):
#         """Compute comprehensive statistics for difficulty assessment"""
#         if len(self.all_triples) == 0:
#             self.entity_freq = defaultdict(int)
#             self.relation_freq = defaultdict(int)
#             self.entity_degree = defaultdict(set)
#             return
            
#         # Entity and relation frequencies
#         self.entity_freq = defaultdict(int)
#         self.relation_freq = defaultdict(int)
#         self.entity_degree = defaultdict(set)
        
#         for triple in self.all_triples:
#             s, r, o = triple
#             self.entity_freq[s] += 1
#             self.entity_freq[o] += 1
#             self.relation_freq[r] += 1
#             # Track entity degrees (number of unique relations)
#             self.entity_degree[s].add(r)
#             self.entity_degree[o].add(r)
        
#         # Compute distribution statistics for adaptive difficulty
#         entity_freqs = list(self.entity_freq.values())
#         self.entity_freq_std = np.std(entity_freqs) if entity_freqs else 0
#         self.entity_freq_mean = np.mean(entity_freqs) if entity_freqs else 0
        
#         relation_freqs = list(self.relation_freq.values())
#         self.relation_freq_std = np.std(relation_freqs) if relation_freqs else 0
#         self.relation_freq_mean = np.mean(relation_freqs) if relation_freqs else 0
        
#     def compute_difficulty_scores(self):
#         """
#         Compute difficulty scores using multiple metrics.
#         Combines temporal, frequency, and structural complexity.
#         """
#         difficulty_scores = []
        
#         for t, snap in enumerate(self.train_list):
#             if len(snap) == 0:
#                 difficulty_scores.append(0.5)  # Neutral difficulty for empty snapshots
#                 continue
            
#             # 1. Temporal difficulty (later = more complex due to accumulated history)
#             temporal_score = t / len(self.train_list)
            
#             # 2. Frequency-based difficulty (rare entities/relations = harder)
#             frequency_scores = []
#             for triple in snap:
#                 s, r, o = triple
#                 s_freq = self.entity_freq.get(s, 1)
#                 o_freq = self.entity_freq.get(o, 1)
#                 r_freq = self.relation_freq.get(r, 1)
                
#                 # Inverse frequency score (lower frequency = higher difficulty)
#                 entity_rarity = 2.0 / (s_freq + o_freq)
#                 relation_rarity = 1.0 / r_freq
#                 frequency_scores.append(entity_rarity + relation_rarity * 0.5)
            
#             frequency_score = np.mean(frequency_scores)
            
#             # 3. Structural complexity (entity degree diversity)
#             degree_scores = []
#             for triple in snap:
#                 s, r, o = triple
#                 s_degree = len(self.entity_degree.get(s, set()))
#                 o_degree = len(self.entity_degree.get(o, set()))
#                 degree_scores.append((s_degree + o_degree) / 2.0)
            
#             degree_score = np.mean(degree_scores) if degree_scores else 0
#             # Normalize degree score
#             max_degree = max(len(degrees) for degrees in self.entity_degree.values()) if self.entity_degree else 1
#             degree_score = degree_score / max(max_degree, 1)
            
#             # 4. Snapshot size complexity (larger snapshots = potentially harder)
#             size_score = len(snap) / max(len(s) for s in self.train_list)
            
#             # Combine all difficulty metrics with adaptive weights
#             # Higher weight on frequency for skewed datasets, temporal for regular datasets
#             freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
#             temporal_weight = 0.8 - freq_weight
            
#             combined_score = (
#                 temporal_weight * temporal_score +
#                 freq_weight * frequency_score +
#                 0.15 * degree_score +
#                 0.05 * size_score
#             )
            
#             difficulty_scores.append(combined_score)
        
#         return np.array(difficulty_scores)

def get_curriculum_samples(train_list, difficulty_scores, current_ratio):
    """
    Get training samples using a general curriculum strategy.
    Balances temporal order with difficulty for optimal learning progression.
    """
    n_samples = max(1, min(int(len(train_list) * current_ratio), len(train_list)))
    
    # Combine temporal order with difficulty-based selection
    # This works well for both ICEWS (temporal patterns) and GDELT (frequency skew)
    
    # Create temporal preference (earlier samples preferred)
    temporal_preference = np.arange(len(train_list)) / len(train_list)
    
    # Combine temporal and difficulty scores
    # Early in curriculum: prefer easy + early samples
    # Later in curriculum: include harder + later samples
    temporal_weight = max(0.3, 1.0 - current_ratio)  # Decrease temporal weight as curriculum progresses
    difficulty_weight = 1.0 - temporal_weight
    
    combined_scores = (
        temporal_weight * temporal_preference +
        difficulty_weight * difficulty_scores
    )
    
    # Select samples with combined scoring
    selected_indices = np.argsort(combined_scores)[:n_samples]
    
    # Ensure we always include some early temporal samples for context
    min_temporal_samples = max(1, min(n_samples // 4, 5))
    early_samples = list(range(1, min(min_temporal_samples + 1, len(train_list))))
    
    # Combine and deduplicate
    selected_indices = sorted(list(set(selected_indices) | set(early_samples)))
    
    return selected_indices[:n_samples]  # Ensure we don't exceed target sample count

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, 
         all_ans_r_list, model_name, static_graph, mode):
    """
    Modified test function with difficulty tracking
    
    CHANGES:
    - Added difficulty_scores parameter
    - Added dataset_name parameter
    - Added PerformanceTracker initialization
    - Added per-snapshot performance tracking
    - Added difficulty vs performance plotting at the end
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    ranks_raw_r_inv, ranks_filter_r_inv, mrr_raw_list_r_inv, mrr_filter_list_r_inv = [], [], [], []
    
    # NEW: Initialize performance tracker
    performance_tracker = PerformanceTracker(len(test_list))

    idx = 0
    if mode == "test":
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    sr_to_sro = np.load('../data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples = test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        
        que_pair = e2r(test_snap, num_rels)
        que_pair_inv = e2r(inverse_triples, num_rels)
        sub_snap, sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap, num_nodes, num_rels, use_cuda, args.gpu)

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
        
        test_triples, final_score = model.predict(que_pair, sub_snap, time_idx, history_glist, num_rels, static_graph, test_triples_input, use_cuda)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        
        # NEW: Calculate hits@k for this snapshot
        def calculate_hits(ranks):
            hits = []
            for k in [1, 3, 10]:
                hits.append((ranks <= k).float().mean().item())
            return hits
        
        hits_filter = calculate_hits(rank_filter)
        hits_filter_inv = calculate_hits(rank_filter_inv)
        
        # NEW: Average metrics for this snapshot
        avg_mrr = (mrr_filter_snap + mrr_filter_snap_inv) / 2
        avg_hits = [(hits_filter[i] + hits_filter_inv[i]) / 2 for i in range(len(hits_filter))]
        
        # NEW: Track performance for this snapshot
        performance_tracker.update(time_idx, avg_mrr, avg_hits)
        
        # Original code continues...
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
        
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        
        idx += 1

    # Original metric calculations
    mrr_raw, hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv, hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv, hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    
    all_mrr_raw = (mrr_raw + mrr_raw_inv) / 2
    all_mrr_filter = (mrr_filter + mrr_filter_inv) / 2
    all_hit_raw, all_hit_filter = [], []
    
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id] + hit_raw_inv[hit_id]) / 2)
        all_hit_filter.append((hit_filter[hit_id] + hit_filter_inv[hit_id]) / 2)
    
    print("(all_raw) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
        all_mrr_raw.item(), all_hit_raw[0], all_hit_raw[1], all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
        all_mrr_filter.item(), all_hit_filter[0], all_hit_filter[1], all_hit_filter[2]))
    
    test_difficulty_scores = compute_test_difficulty_scores(
        train_list=history_list,  # Use training data for statistics
        test_list=test_list,
        num_rels=num_rels
    )
    
    # Analyze difficulty vs performance
    correlations = analyze_test_difficulty_vs_performance(
        test_list=test_list,
        test_difficulty_scores=test_difficulty_scores,
        test_mrr=test_mrr_per_snapshot,
        test_h1=test_h1_per_snapshot,
        test_h10=test_h10_per_snapshot
    )
    
    # Original CSV writing code
    if mode == "test":
        filename = '../result/'+ args.dataset + ".csv"
        if os.path.isfile(filename) == False:
            with open (filename,'w', newline='') as f:
                fieldnames=['encoder','opn','pre_type','use_static','use_cl','gpu','datetime','pre_weight',
                            'train_len','test_len','temperature','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        with open (filename,'a', newline='') as f:
            writer = csv.writer(f)
            row={'encoder':args.encoder,'opn':args.opn,'pre_type':args.pre_type,'use_static':args.add_static_graph,'use_cl':args.use_cl,'gpu':args.gpu,'datetime':datetime.now(),'pre_weight':args.pre_weight,
                'train_len':args.train_history_len,'test_len':args.test_history_len,'temperature':args.temperature,'lr':args.lr,'n_hidden':args.n_hidden,
                'filter_MRR':float(mrr_filter),'filter_H@1':hit_filter[0],'filter_H@3':hit_filter[1],'filter_H@10':hit_filter[2],
                'filter_inv_MRR':float(mrr_filter_inv),'filter_inv_H@1':hit_filter_inv[0],'filter_inv_H@3':hit_filter_inv[1],'filter_inv_H@10':hit_filter_inv[2],
                'all_MRR':all_mrr_raw.item(),'all_H@1':all_hit_raw[0],'all_H@3':all_hit_raw[1],'all_H@10':all_hit_raw[2],
                'filter_all_MRR':all_mrr_filter.item(),'filter_all_H@1':all_hit_filter[0],'filter_all_H@3':all_hit_filter[1],'filter_all_H@10':all_hit_filter[2]}
            writer.writerow(row.values())
            
    return all_mrr_raw, all_mrr_filter, correlations
    

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    global mrr_raw, mrr_filter
    mrr_raw, mrr_filter = [], []
    print('date time now is',datetime.now())
    """
    General curriculum learning experiment that adapts to any temporal KG dataset.
    """
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    print(f"Train snapshots: {len(train_list)}")  # Should be ~300
    print(f"Test snapshots: {len(test_list)}")    # Should be ~30
    
    # Compute difficulty for TEST set (not training!)
    test_difficulty_scores = compute_test_difficulty_scores(
        train_list=train_list,
        test_list=test_list,
        num_rels=data.num_rels
    )

    # Initialize general curriculum learning components
    use_curriculum = getattr(args, 'use_curriculum', True)
    # difficulty_scores = None  # NEW: Initialize this variable
    
    if use_curriculum:
        curriculum_scheduler = GeneralCurriculumScheduler(
            total_epochs=args.n_epochs,
            warmup_ratio=getattr(args, 'curriculum_warmup_ratio', 0.4),
            strategy=getattr(args, 'curriculum_strategy', 'adaptive')
        )
        
        # difficulty_analyzer = GeneralDifficultyAnalyzer(train_list)
        difficulty_analyzer = GeneralDifficultyAnalyzer(train_list, ablation_mode='all')
        difficulty_scores = difficulty_analyzer.compute_difficulty_scores()
        
        print(f"Curriculum Learning Enabled:")
        print(f"  Strategy: {curriculum_scheduler.strategy}")
        print(f"  Warmup Epochs: {curriculum_scheduler.warmup_epochs}")
        print(f"  Start Ratio: {curriculum_scheduler.start_ratio}")
        print(f"  Data Statistics:")
        print(f"    Entity freq std/mean: {difficulty_analyzer.entity_freq_std:.2f}/{difficulty_analyzer.entity_freq_mean:.2f}")
        print(f"    Relation freq std/mean: {difficulty_analyzer.relation_freq_std:.2f}/{difficulty_analyzer.relation_freq_mean:.2f}")
        print(f" use contrastive ? : ", args.use_cl)
    else:
        print("Curriculum Learning Disabled - Using standard training")
        print(f" use contrastive ? : ", args.use_cl)

    # Load answer lists for evaluation
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature, args.pre_weight, 
                args.use_cl, args.pre_type, args.n_hidden, args.encoder, str(time.time()))
    model_state_file = '../models/' + 'esh' 
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # Static graph setup (if applicable)
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # Create model
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        pre_weight=args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type=args.pre_type,
                        use_cl=args.use_cl,
                        temperature=args.temperature,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu=args.gpu,
                        analysis=args.run_analysis,
                        cl_approach = args.cl_approach)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    print(f"Model created with curriculum learning approach: {args.cl_approach}")

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(model,
                                train_list+valid_list, 
                                test_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                all_ans_list_test, 
                                all_ans_list_r_test, 
                                model_state_file, 
                                static_graph, 
                                "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        if use_curriculum:
            print("Using general curriculum learning")
        
        best_mrr = 0
        patience_counter = 0
        patience_limit = 5
        avgloss = []
        
        # Training loop
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            # Get current curriculum samples
            if use_curriculum:
                current_ratio = curriculum_scheduler.get_current_ratio(epoch)
                selected_indices = get_curriculum_samples(
                    train_list, 
                    difficulty_scores, 
                    current_ratio
                )
                print(f"Epoch {epoch}: Using {len(selected_indices)}/{len(train_list)} samples (ratio: {current_ratio:.3f})")
            else:
                # Use all samples without curriculum
                selected_indices = list(range(1, len(train_list)))  # Skip index 0
                current_ratio = 1.0

            # Shuffle selected indices to avoid order bias within curriculum
            if use_curriculum and epoch > 0:
                random.shuffle(selected_indices)
                

            for train_sample_num in tqdm(selected_indices):
                if train_sample_num == 0: 
                    continue
                    
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # Load subgraph data with fallback
                try:
                    # subgraph_arr = np.load('../data/{}/his_graph_for_new/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                    subgraph_arr = np.load('../data/{}/his_graph_for/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                    # subgraph_arr_inv = np.load('../data/{}/his_graph_inv_new/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                    subgraph_arr_inv = np.load('../data/{}/his_graph_inv/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                except FileNotFoundError:
                    # Fallback: create subgraph from available history
                    if len(input_list) > 0:
                        # Use recent history for subgraph construction
                        history_window = min(3, len(input_list))
                        subgraph_arr = np.concatenate(input_list[-history_window:])
                        subgraph_arr_inv = subgraph_arr[:, [2, 1, 0]]
                        subgraph_arr_inv[:, 1] = subgraph_arr_inv[:, 1] + num_rels
                    else:
                        # Skip this sample if no history available
                        continue
                
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair = e2r(output[0], num_rels)
                que_pair_inv = e2r(inverse_triples, num_rels)
                
                # Generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda() if use_cuda else torch.from_numpy(output[0]).long()
                inverse_triples_tensor = torch.from_numpy(inverse_triples).long().cuda() if use_cuda else torch.from_numpy(inverse_triples).long()
                
                # Forward and backward pass for both directions
                for direction in range(2): 
                    try:
                        if direction % 2 == 0: 
                            loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair, subg_snap, train_sample_num, history_glist, triples, static_graph, use_cuda)
                        else:
                            loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num, history_glist, inverse_triples_tensor, static_graph, use_cuda)

                        loss = loss_e + loss_static + loss_cl
                    
                        losses.append(loss.item())
                        losses_e.append(loss_e.item())
                        losses_r.append(loss_r.item())
                        losses_static.append(loss_static.item())
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                    except RuntimeError as e:
                        # Handle potential GPU memory issues gracefully
                        print(f"Warning: Skipping sample {train_sample_num} due to error: {e}")
                        if use_cuda:
                            torch.cuda.empty_cache()
                        continue
                if use_cuda:
                    torch.cuda.empty_cache()    
            # Record average loss
            avg_loss = np.mean(losses) if losses else 0.0
            avgloss.append(avg_loss)
            
            # Print epoch statistics
            if use_curriculum:
                curriculum_info = f"| Curriculum Ratio: {current_ratio:.3f}"
                # Show curriculum progress
                if epoch < curriculum_scheduler.warmup_epochs:
                    progress_pct = (epoch / curriculum_scheduler.warmup_epochs) * 100
                    curriculum_info += f" | Curriculum Progress: {progress_pct:.1f}%"
            else:
                curriculum_info = ""
                
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} {} | Best MRR {:.4f}"
                  .format(epoch, avg_loss, 
                         np.mean(losses_e) if losses_e else 0.0, 
                         np.mean(losses_r) if losses_r else 0.0, 
                         np.mean(losses_static) if losses_static else 0.0, 
                         curriculum_info, best_mrr))
            if use_cuda:
                torch.cuda.empty_cache()
            # Validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, 
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph, 
                                    mode="train",
                                    difficulty_scores=None,  # NEW: No analysis during training
                                    dataset_name=args.dataset)  # NEW: Pass dataset name)
                
                # Early stopping with patience
                if not args.relation_evaluation:
                    if mrr_filter < best_mrr:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{patience_limit}")
                        if patience_counter >= patience_limit:
                            print(f"Early stopping triggered after {patience_limit} epochs without improvement")
                            break
                    else:
                        patience_counter = 0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                        print(f"New best MRR: {best_mrr:.4f} - Model saved")
                        
            # Clear GPU cache to prevent memory issues
            if use_cuda:
                torch.cuda.empty_cache()
        np.savetxt('lossval.txt', avgloss)
        plt.title('Plot of training loss')
        plt.plot(avgloss)
        plt.savefig('lossfig.png')
        # # Save loss plot
        # if avgloss:
        #     np.savetxt('lossval.txt', avgloss)
        #     plt.figure(figsize=(12, 8))
            
        #     # Create subplot for loss and curriculum progression
        #     if use_curriculum:
        #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
        #         # Loss plot
        #         ax1.plot(avgloss, 'b-', linewidth=2)
        #         ax1.set_title(f'Training Loss - {args.dataset.upper()} with General Curriculum Learning')
        #         ax1.set_xlabel('Epoch')
        #         ax1.set_ylabel('Loss')
        #         ax1.grid(True, alpha=0.3)
                
        #         # Curriculum progression plot
        #         curriculum_ratios = [curriculum_scheduler.get_current_ratio(e) for e in range(len(avgloss))]
        #         ax2.plot(curriculum_ratios, 'r-', linewidth=2)
        #         ax2.axvline(x=curriculum_scheduler.warmup_epochs, color='gray', linestyle='--', alpha=0.7, label='Curriculum Complete')
        #         ax2.set_title('Curriculum Learning Progression')
        #         ax2.set_xlabel('Epoch')
        #         ax2.set_ylabel('Data Ratio Used')
        #         ax2.set_ylim(0, 1.1)
        #         ax2.grid(True, alpha=0.3)
        #         ax2.legend()
                
        #         plt.tight_layout()
        #     else:
        #         plt.plot(avgloss, 'b-', linewidth=2)
        #         plt.title(f'Training Loss - {args.dataset.upper()}')
        #         plt.xlabel('Epoch')
        #         plt.ylabel('Loss')
        #         plt.grid(True, alpha=0.3)
            
        #     plt.savefig('lossfig.png', dpi=300, bbox_inches='tight')
        #     plt.close()

        # Final testing
        print("\n" + "="*60)
        print("Starting final evaluation on test set...")
        if use_curriculum:
            print(f"Curriculum learning completed after {curriculum_scheduler.warmup_epochs} warmup epochs")
            print(f"Final strategy used: {curriculum_scheduler.strategy}")
        print("="*60)
        
        mrr_raw, mrr_filter = test(model, 
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph, 
                            mode="test",
                            difficulty_scores=difficulty_scores,  # NEW: Pass difficulty scores!
                            dataset_name=args.dataset)  # NEW: Pass dataset name!)
                            
        print("="*60)
        print("Training completed successfully!")
        if use_curriculum:
            print(f"Curriculum learning provided structured learning progression")
            print(f"Dataset characteristics automatically detected and handled")
        print("="*60)
        print('date time now is',datetime.now())
        
    return mrr_raw, mrr_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kgta')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="short",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=False,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    
    # #config for curriculum learning
    # parser.add_argument("--curriculum_strategy", type=str, default="linear", # 'linear', 'exponential', 'step', 'cosine'
    #                     help="curriculum strategy")
    # parser.add_argument("--difficulty_metric", type=str, default="frequency", # 'frequency', 'rarity', 'temporal_distance', 'degree'
    #                     help="difficulty metric")
    # parser.add_argument("--curriculum_start_ratio", type=float, default=0.1, 
    #                     help="curriculum start ratio")
    # parser.add_argument("--curriculum_end_ratio", type=float, default=1.0, 
    #                     help="curriculum end ratio")
    # parser.add_argument("--curriculum_warmup_epochs", type=int, default=5,  #Reach full curriculum in 5 epochs
    #                     help="curriculum epochs")
    # parser.add_argument("--curriculum_sample_strategy", type=str, default="easy_first",  # 'easy_first', 'hard_first', 'mixed'
    #                     help="curriculum strategy")

    
    parser.add_argument("--cl_approach", type=str, default="original", 
                        choices=["original", "cosine_positive", "barlow_twins", "mse_positive", "temporal_consistency","laplace"],
                        help="positive pairs only approach")
 
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")


    args = parser.parse_args()
    print(args)
    args.__dict__["test_history_len"] = args.__dict__["train_history_len"]

    run_experiment(args)