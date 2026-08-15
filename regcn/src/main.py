"""
main_curriculum.py

RE-GCN (main.py) + STRIDE curriculum learning, ported from the LogCL-based
curriculum script.

Why this port is simpler than the TIRGN one:
  - RE-GCN's `split_by_time` (rgcn/utils.py, document 14) truncates every
    triple to `train[:3]` -- no timestamp column survives into the
    snapshots, so the difficulty analyzer needs no `triple[:3]` slicing
    (triples are already 3 columns, same as LogCL's).
  - `split_by_time` returns a plain `snapshot_list` (not a `(snapshots,
    times)` tuple like TIRGN's), matching what GeneralDifficultyAnalyzer /
    get_curriculum_samples already expect.
  - `RecurrentRGCN.get_loss(glist, triples, static_graph, use_cuda)` and
    `.predict(test_graph, num_rels, static_graph, test_triplets, use_cuda)`
    take no `que_pair`/`sub_graph` (that's LogCL's contrastive-learning
    machinery) and no sparse history-vocab lookup (that's TIRGN's). So the
    curriculum-modified training loop only has to change *which* snapshot
    indices get iterated over each epoch -- nothing else about the
    forward/backward pass changes at all.
  - There's no `use_cl` flag here either (no contrastive learning in
    RE-GCN), so curriculum learning is again the only scheduling axis.

One thing NOT changed: this RE-GCN's `utils.stat_ranks()` only returns a
scalar MRR (not `(mrr, hit_result)` like TIRGN's version), so `test()` here
still only returns 4 MRR values -- no Hits@k are available to report/log
without also changing `rgcn/utils.py`, which this file leaves untouched.
"""

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
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list

import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
#  Curriculum learning components (identical to the LogCL version --
#  RE-GCN triples are already [s, r, o], no slicing changes needed)
# =====================================================================

class GeneralCurriculumScheduler:
    """
    General curriculum learning scheduler that adapts to any temporal
    knowledge graph dataset. Pure epoch/ratio bookkeeping -- unchanged.
    """

    def __init__(self, total_epochs=50, warmup_ratio=0.4, strategy='adaptive'):
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.strategy = strategy
        self.start_ratio = 0.2  # Start with 20% of data

    def get_current_ratio(self, epoch):
        if epoch >= self.warmup_epochs:
            return 1.0

        progress = epoch / self.warmup_epochs

        if self.strategy == 'adaptive':
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (
                0.5 * (1 - np.cos(progress * np.pi)) + 0.3 * progress
            )
        elif self.strategy == 'cosine':
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (1 - np.cos(progress * np.pi)) / 2
        else:
            ratio = self.start_ratio + (1.0 - self.start_ratio) * progress

        return min(ratio, 1.0)


class GeneralDifficultyAnalyzer:
    """
    General difficulty analyzer that works for any TKG dataset.
    RE-GCN's snapshots are [s, r, o] triples (3 columns) exactly like
    LogCL's, so triple unpacking here is the plain `s, r, o = triple` --
    no `[:3]` slicing needed (that was TIRGN-specific).
    """

    def __init__(self, train_list, ablation_mode='all'):
        self.train_list = train_list
        self.all_triples = np.concatenate(train_list) if train_list else np.array([])
        self.ablation_mode = ablation_mode
        self._compute_statistics()

    def _compute_statistics(self):
        if len(self.all_triples) == 0:
            self.entity_freq = defaultdict(int)
            self.relation_freq = defaultdict(int)
            self.entity_degree = defaultdict(set)
            return

        self.entity_freq = defaultdict(int)
        self.relation_freq = defaultdict(int)
        self.entity_degree = defaultdict(set)

        for triple in self.all_triples:
            s, r, o = triple
            self.entity_freq[s] += 1
            self.entity_freq[o] += 1
            self.relation_freq[r] += 1
            self.entity_degree[s].add(r)
            self.entity_degree[o].add(r)

        entity_freqs = list(self.entity_freq.values())
        self.entity_freq_std = np.std(entity_freqs) if entity_freqs else 0
        self.entity_freq_mean = np.mean(entity_freqs) if entity_freqs else 0

        relation_freqs = list(self.relation_freq.values())
        self.relation_freq_std = np.std(relation_freqs) if relation_freqs else 0
        self.relation_freq_mean = np.mean(relation_freqs) if relation_freqs else 0

    def compute_difficulty_scores(self):
        """Compute difficulty scores using multiple metrics, normalized to [0, 1]."""
        difficulty_scores = []

        all_frequency_scores = []
        all_degree_scores = []
        all_size_scores = []

        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                continue

            if self.ablation_mode in ['all', 'first_3', 'first_2']:
                snap_freq_scores = []
                for triple in snap:
                    s, r, o = triple
                    s_freq = self.entity_freq.get(s, 1)
                    o_freq = self.entity_freq.get(o, 1)
                    r_freq = self.relation_freq.get(r, 1)

                    entity_rarity = 2.0 / (s_freq + o_freq)
                    relation_rarity = 1.0 / r_freq
                    snap_freq_scores.append(entity_rarity + relation_rarity * 0.5)

                if snap_freq_scores:
                    all_frequency_scores.append(np.mean(snap_freq_scores))

            if self.ablation_mode in ['all', 'first_3']:
                snap_degree_scores = []
                for triple in snap:
                    s, r, o = triple
                    s_degree = len(self.entity_degree.get(s, set()))
                    o_degree = len(self.entity_degree.get(o, set()))
                    snap_degree_scores.append((s_degree + o_degree) / 2.0)

                if snap_degree_scores:
                    all_degree_scores.append(np.mean(snap_degree_scores))

            if self.ablation_mode == 'all':
                all_size_scores.append(len(snap))

        freq_min = min(all_frequency_scores) if all_frequency_scores else 0
        freq_max = max(all_frequency_scores) if all_frequency_scores else 1
        freq_range = max(freq_max - freq_min, 1e-8)

        degree_min = min(all_degree_scores) if all_degree_scores else 0
        degree_max = max(all_degree_scores) if all_degree_scores else 1
        degree_range = max(degree_max - degree_min, 1e-8)

        size_min = min(all_size_scores) if all_size_scores else 0
        size_max = max(all_size_scores) if all_size_scores else 1
        size_range = max(size_max - size_min, 1e-8)

        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                difficulty_scores.append(0.5)
                continue

            if self.ablation_mode == 'none':
                difficulty_scores.append(0.5)
                continue

            temporal_score = t / max(len(self.train_list) - 1, 1)

            frequency_score = 0.0
            if self.ablation_mode in ['all', 'first_3', 'first_2']:
                frequency_scores_snap = []
                for triple in snap:
                    s, r, o = triple
                    s_freq = self.entity_freq.get(s, 1)
                    o_freq = self.entity_freq.get(o, 1)
                    r_freq = self.relation_freq.get(r, 1)

                    entity_rarity = 2.0 / (s_freq + o_freq)
                    relation_rarity = 1.0 / r_freq
                    raw_score = entity_rarity + relation_rarity * 0.5

                    normalized_score = (raw_score - freq_min) / freq_range
                    frequency_scores_snap.append(normalized_score)

                frequency_score = np.mean(frequency_scores_snap)

            degree_score = 0.0
            if self.ablation_mode in ['all', 'first_3']:
                degree_scores_snap = []
                for triple in snap:
                    s, r, o = triple
                    s_degree = len(self.entity_degree.get(s, set()))
                    o_degree = len(self.entity_degree.get(o, set()))
                    raw_degree = (s_degree + o_degree) / 2.0

                    normalized_degree = (raw_degree - degree_min) / degree_range
                    degree_scores_snap.append(normalized_degree)

                degree_score = np.mean(degree_scores_snap) if degree_scores_snap else 0

            size_score = 0.0
            if self.ablation_mode == 'all':
                size_score = (len(snap) - size_min) / size_range

            if self.ablation_mode == 'first_1':
                combined_score = temporal_score

            elif self.ablation_mode == 'first_2':
                freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
                temporal_weight = 1.0 - freq_weight

                combined_score = (
                    temporal_weight * temporal_score +
                    freq_weight * frequency_score
                )

            elif self.ablation_mode == 'first_3':
                freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
                temporal_weight = 0.85 - freq_weight

                combined_score = (
                    temporal_weight * temporal_score +
                    freq_weight * frequency_score +
                    0.15 * degree_score
                )

            else:  # 'all'
                freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
                temporal_weight = 0.8 - freq_weight

                combined_score = (
                    temporal_weight * temporal_score +
                    freq_weight * frequency_score +
                    0.15 * degree_score +
                    0.05 * size_score
                )

            combined_score = np.clip(combined_score, 0.0, 1.0)
            difficulty_scores.append(combined_score)

        scores_array = np.array(difficulty_scores)

        if len(scores_array) > 0:
            scores_min = scores_array.min()
            scores_max = scores_array.max()
            if scores_max > scores_min:
                scores_array = (scores_array - scores_min) / (scores_max - scores_min)

        return scores_array


def get_curriculum_samples(train_list, difficulty_scores, current_ratio):
    """Unchanged -- purely index-level logic."""
    n_samples = max(1, min(int(len(train_list) * current_ratio), len(train_list)))

    temporal_preference = np.arange(len(train_list)) / len(train_list)

    temporal_weight = max(0.3, 1.0 - current_ratio)
    difficulty_weight = 1.0 - temporal_weight

    combined_scores = (
        temporal_weight * temporal_preference +
        difficulty_weight * difficulty_scores
    )

    ranked_indices = np.argsort(combined_scores)  # easiest -> hardest, full ranking
    selected_set = set(ranked_indices[:n_samples].tolist())

    min_temporal_samples = max(1, min(n_samples // 4, 5))
    early_samples = set(range(1, min(min_temporal_samples + 1, len(train_list))))

    # Union in the forced early samples, then re-rank the UNION by
    # combined_scores (not chronological index) before truncating back to
    # n_samples, so the difficulty ranking survives the union step.
    union_indices = selected_set | early_samples
    union_sorted_by_difficulty = sorted(union_indices, key=lambda idx: combined_scores[idx])

    return union_sorted_by_difficulty[:n_samples]


class AdaptiveWeightPrinter:
    """Prints the adaptive weight values calculated from dataset statistics."""

    def print_adaptive_weights(self, difficulty_analyzer, ablation_mode='all', dataset_name=''):
        print("\n" + "=" * 60)
        print(f"ADAPTIVE WEIGHT VALUES - {dataset_name}")
        print("=" * 60)

        if ablation_mode in ('first_2', 'first_3', 'all'):
            freq_weight = min(0.6, 0.3 + difficulty_analyzer.entity_freq_std /
                               max(difficulty_analyzer.entity_freq_mean, 1) * 0.1)

            if ablation_mode == 'first_2':
                temporal_weight = 1.0 - freq_weight
                degree_weight = 0.0
                size_weight = 0.0
            elif ablation_mode == 'first_3':
                temporal_weight = 0.85 - freq_weight
                degree_weight = 0.15
                size_weight = 0.0
            else:  # 'all'
                temporal_weight = 0.8 - freq_weight
                degree_weight = 0.15
                size_weight = 0.05
        elif ablation_mode == 'first_1':
            temporal_weight, freq_weight, degree_weight, size_weight = 1.0, 0.0, 0.0, 0.0
        else:  # 'none'
            temporal_weight, freq_weight, degree_weight, size_weight = 0.5, 0.0, 0.0, 0.0

        print(f"\nAblation Mode: {ablation_mode}")
        print(f"\nDataset Statistics:")
        print(f"  Entity Frequency Mean: {difficulty_analyzer.entity_freq_mean:.4f}")
        print(f"  Entity Frequency Std:  {difficulty_analyzer.entity_freq_std:.4f}")
        print(f"  Std/Mean Ratio:        "
              f"{difficulty_analyzer.entity_freq_std / max(difficulty_analyzer.entity_freq_mean, 1):.4f}")

        print(f"\nCalculated Adaptive Weights:")
        print(f"  Temporal Weight:  {temporal_weight:.4f}")
        print(f"  Frequency Weight: {freq_weight:.4f}")
        print(f"  Degree Weight:    {degree_weight:.4f}")
        print(f"  Size Weight:      {size_weight:.4f}")
        print(f"  Total:            {temporal_weight + freq_weight + degree_weight + size_weight:.4f}")
        print("=" * 60 + "\n")

        return {
            'temporal_weight': temporal_weight,
            'freq_weight': freq_weight,
            'degree_weight': degree_weight,
            'size_weight': size_weight,
        }


class DifficultyVisualizer:
    """Same as LogCL's version -- RE-GCN triples are already 3-column."""

    def __init__(self, difficulty_analyzer, train_list):
        self.analyzer = difficulty_analyzer
        self.train_list = train_list
        self.difficulty_scores = None
        self.component_scores = None

    def compute_component_scores(self):
        temporal_scores, frequency_scores, degree_scores, size_scores = [], [], [], []
        raw_frequency_scores, raw_degree_scores, raw_size_scores = [], [], []

        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                continue

            freq_vals = []
            for triple in snap:
                s, r, o = triple
                s_freq = self.analyzer.entity_freq.get(s, 1)
                o_freq = self.analyzer.entity_freq.get(o, 1)
                r_freq = self.analyzer.relation_freq.get(r, 1)
                entity_rarity = 2.0 / (s_freq + o_freq)
                relation_rarity = 1.0 / r_freq
                freq_vals.append(entity_rarity + relation_rarity * 0.5)
            if freq_vals:
                raw_frequency_scores.append(np.mean(freq_vals))

            deg_vals = []
            for triple in snap:
                s, r, o = triple
                s_degree = len(self.analyzer.entity_degree.get(s, set()))
                o_degree = len(self.analyzer.entity_degree.get(o, set()))
                deg_vals.append((s_degree + o_degree) / 2.0)
            if deg_vals:
                raw_degree_scores.append(np.mean(deg_vals))

            raw_size_scores.append(len(snap))

        freq_min = min(raw_frequency_scores) if raw_frequency_scores else 0
        freq_max = max(raw_frequency_scores) if raw_frequency_scores else 1
        freq_range = max(freq_max - freq_min, 1e-8)

        degree_min = min(raw_degree_scores) if raw_degree_scores else 0
        degree_max = max(raw_degree_scores) if raw_degree_scores else 1
        degree_range = max(degree_max - degree_min, 1e-8)

        size_min = min(raw_size_scores) if raw_size_scores else 0
        size_max = max(raw_size_scores) if raw_size_scores else 1
        size_range = max(size_max - size_min, 1e-8)

        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                temporal_scores.append(0.5)
                frequency_scores.append(0.5)
                degree_scores.append(0.5)
                size_scores.append(0.5)
                continue

            temporal_score = t / max(len(self.train_list) - 1, 1)
            temporal_scores.append(temporal_score)

            freq_vals = []
            for triple in snap:
                s, r, o = triple
                s_freq = self.analyzer.entity_freq.get(s, 1)
                o_freq = self.analyzer.entity_freq.get(o, 1)
                r_freq = self.analyzer.relation_freq.get(r, 1)
                entity_rarity = 2.0 / (s_freq + o_freq)
                relation_rarity = 1.0 / r_freq
                freq_vals.append(entity_rarity + relation_rarity * 0.5)
            raw_freq_score = np.mean(freq_vals) if freq_vals else 0
            frequency_scores.append((raw_freq_score - freq_min) / freq_range)

            deg_vals = []
            for triple in snap:
                s, r, o = triple
                s_degree = len(self.analyzer.entity_degree.get(s, set()))
                o_degree = len(self.analyzer.entity_degree.get(o, set()))
                deg_vals.append((s_degree + o_degree) / 2.0)
            raw_degree_score = np.mean(deg_vals) if deg_vals else 0
            degree_scores.append((raw_degree_score - degree_min) / degree_range)

            raw_size_score = len(snap)
            size_scores.append((raw_size_score - size_min) / size_range)

        self.component_scores = {
            'Temp': np.array(temporal_scores),
            'Freq': np.array(frequency_scores),
            'Struct': np.array(degree_scores),
            'Size': np.array(size_scores),
        }

        print("\nComponent Score Ranges (After Normalization):")
        for component, scores in self.component_scores.items():
            non_empty = scores[scores != 0.5]
            if len(non_empty) > 0:
                print(f"  {component}: [{non_empty.min():.4f}, {non_empty.max():.4f}] "
                      f"(mean={non_empty.mean():.4f}, std={non_empty.std():.4f})")

        return self.component_scores

    def plot_combined_difficulty(self, save_path='combined_difficulty.png', figsize=(14, 6), dataset_name='REGCN'):
        if self.component_scores is None:
            self.compute_component_scores()

        if self.analyzer.ablation_mode == 'all':
            freq_weight = min(0.6, 0.3 + self.analyzer.entity_freq_std / max(self.analyzer.entity_freq_mean, 1) * 0.1)
            temporal_weight = 0.8 - freq_weight

            combined = (
                temporal_weight * self.component_scores['Temp'] +
                freq_weight * self.component_scores['Freq'] +
                0.15 * self.component_scores['Struct'] +
                0.05 * self.component_scores['Size']
            )
        else:
            combined = np.mean([self.component_scores[name] for name in ['Temp', 'Freq', 'Struct', 'Size']], axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        snapshot_indices = np.arange(len(self.train_list))
        colors = ["#1F77B4", "#C44E52", "#FFA600", "#2F9E44"]
        component_names = ['Temp', 'Freq', 'Struct', 'Size']

        for name, color in zip(component_names, colors):
            ax1.plot(snapshot_indices, self.component_scores[name], color=color, linewidth=3, alpha=0.7, label=name)
        ax1.plot(snapshot_indices, combined, color='black', linewidth=3, linestyle='--', label='Composite', alpha=0.9)
        ax1.set_xlabel('Snapshot Index', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Difficulty Score', fontsize=14, fontweight='bold')
        ax1.legend(prop={'size': 11}, loc=2, ncol=3, mode="expand", framealpha=0.5)
        ax1.grid(True, alpha=0.3)

        ax2.hist(combined, bins=25, color='#4C72B0', alpha=0.7, edgecolor='black', density=True)
        try:
            kde = scipy_stats.gaussian_kde(combined)
            x_range = np.linspace(combined.min(), combined.max(), 200)
            ax2.plot(x_range, kde(x_range), color='#2B3A4A', linewidth=3, label='KDE')
        except Exception:
            pass
        mean_val = np.mean(combined)
        ax2.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=3, label=f'Mean: {mean_val:.2f}')
        ax2.set_xlabel('Composite Difficulty Score', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax2.legend(prop={'size': 11})
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Composite Difficulty Score Analysis - {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined difficulty to {save_path}")
        plt.close()
        return combined

    def generate_all_visualizations(self, output_dir='./visualizations/', dataset_name='REGCN'):
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("GENERATING DIFFICULTY LANDSCAPE VISUALIZATIONS")
        print("=" * 60 + "\n")
        self.plot_combined_difficulty(
            save_path=os.path.join(output_dir, 'combined_difficulty.png'), dataset_name=dataset_name)
        print("\nOutput directory: {}\n".format(output_dir))


def visualize_difficulty_landscape(train_list, dataset_name='REGCN'):
    difficulty_analyzer = GeneralDifficultyAnalyzer(train_list, ablation_mode='all')
    visualizer = DifficultyVisualizer(difficulty_analyzer, train_list)
    output_dir = f'./visualizations_{dataset_name}/'
    visualizer.generate_all_visualizations(output_dir=output_dir, dataset_name=dataset_name)
    return visualizer


# =====================================================================
#  RE-GCN test loop (unchanged from the original main.py)
# =====================================================================

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


# =====================================================================
#  run_experiment: RE-GCN training loop + curriculum learning
# =====================================================================

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # ---------------------------------------------------------------
    # Curriculum learning setup (STRIDE, ported from the LogCL script)
    # ---------------------------------------------------------------
    use_curriculum = getattr(args, 'use_curriculum', True)

    print("Generating difficulty landscape visualizations...")
    difficulty_visualizer = visualize_difficulty_landscape(train_list, args.dataset)

    if use_curriculum:
        curriculum_scheduler = GeneralCurriculumScheduler(
            total_epochs=args.n_epochs,
            warmup_ratio=getattr(args, 'curriculum_warmup_ratio', 0.4),
            strategy=getattr(args, 'curriculum_strategy', 'adaptive'),
        )
        difficulty_analyzer = GeneralDifficultyAnalyzer(train_list, ablation_mode='all')
        difficulty_scores = difficulty_analyzer.compute_difficulty_scores()

        print(f"Curriculum Learning Enabled:")
        print(f"  Strategy: {curriculum_scheduler.strategy}")
        print(f"  Warmup Epochs: {curriculum_scheduler.warmup_epochs}")
        print(f"  Start Ratio: {curriculum_scheduler.start_ratio}")
        print(f"  Data Statistics:")
        print(f"    Entity freq std/mean: {difficulty_analyzer.entity_freq_std:.2f}/{difficulty_analyzer.entity_freq_mean:.2f}")
        print(f"    Relation freq std/mean: {difficulty_analyzer.relation_freq_std:.2f}/{difficulty_analyzer.relation_freq_mean:.2f}")

        weight_printer = AdaptiveWeightPrinter()
        weight_printer.print_adaptive_weights(
            difficulty_analyzer=difficulty_analyzer, ablation_mode='all', dataset_name=args.dataset.upper())
    else:
        print("Curriculum Learning Disabled - Using standard uniform training")

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}-cl{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, int(use_curriculum))
    model_state_file = '../models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

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
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
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
            print("Using STRIDE curriculum learning on top of RE-GCN")
        best_mrr = 0
        avgloss = []

        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            # -------------------------------------------------------
            # Curriculum-based snapshot selection (replaces the
            # original `idx = list(range(len(train_list))); shuffle`)
            # -------------------------------------------------------
            if use_curriculum:
                current_ratio = curriculum_scheduler.get_current_ratio(epoch)
                selected_indices = get_curriculum_samples(train_list, difficulty_scores, current_ratio)
                print(f"Epoch {epoch}: Using {len(selected_indices)}/{len(train_list)} snapshots "
                      f"(ratio: {current_ratio:.3f})")
                if epoch > 0:
                    random.shuffle(selected_indices)
                idx = selected_indices
            else:
                idx = [_ for _ in range(len(train_list))]
                random.shuffle(idx)
                current_ratio = 1.0

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = np.mean(losses) if losses else 0.0
            avgloss.append(avg_loss)

            curriculum_info = ""
            if use_curriculum:
                curriculum_info = f"| Curriculum Ratio: {current_ratio:.3f}"
                if epoch < curriculum_scheduler.warmup_epochs:
                    progress_pct = (epoch / curriculum_scheduler.warmup_epochs) * 100
                    curriculum_info += f" | Curriculum Progress: {progress_pct:.1f}%"

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} {} Best MRR {:.4f} | Model {} "
                  .format(epoch, avg_loss,
                          np.mean(losses_e) if losses_e else 0.0,
                          np.mean(losses_r) if losses_r else 0.0,
                          np.mean(losses_static) if losses_static else 0.0,
                          curriculum_info, best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    train_list,
                                                                    valid_list,
                                                                    num_rels,
                                                                    num_nodes,
                                                                    use_cuda,
                                                                    all_ans_list_valid,
                                                                    all_ans_list_r_valid,
                                                                    model_state_file,
                                                                    static_graph,
                                                                    mode="train")

                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        if avgloss:
            np.savetxt('lossval.txt', avgloss)
            plt.figure(figsize=(12, 8))
            if use_curriculum:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                ax1.plot(avgloss, 'b-', linewidth=2)
                ax1.set_title(f'Training Loss - {args.dataset.upper()} (RE-GCN + STRIDE curriculum)')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True, alpha=0.3)

                curriculum_ratios = [curriculum_scheduler.get_current_ratio(e) for e in range(len(avgloss))]
                ax2.plot(curriculum_ratios, 'r-', linewidth=2)
                ax2.axvline(x=curriculum_scheduler.warmup_epochs, color='gray', linestyle='--', alpha=0.7, label='Curriculum Complete')
                ax2.set_title('Curriculum Learning Progression')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Data Ratio Used')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                plt.tight_layout()
            else:
                plt.plot(avgloss, 'b-', linewidth=2)
                plt.title(f'Training Loss - {args.dataset.upper()} (RE-GCN, no curriculum)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
            plt.savefig('lossfig.png', dpi=300, bbox_inches='tight')
            plt.close()

        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            train_list+valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            mode="test")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN + STRIDE curriculum learning')

    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--batch-size", type=int, default=1, help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False, help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False, help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False, help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False, help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10, help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action='store_true', default=False, help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False, help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False, help="save model accordding to the relation evalution")

    parser.add_argument("--weight", type=float, default=1, help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7, help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1, help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10, help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn", help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none", help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False, help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub", help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100, help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100, help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True, help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False, help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False, help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False, help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False, help="do relation prediction")

    parser.add_argument("--n-epochs", type=int, default=500, help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")

    parser.add_argument("--evaluate-every", type=int, default=20, help="perform evaluation every n epochs")

    parser.add_argument("--decoder", type=str, default="convtranse", help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

    parser.add_argument("--train-history-len", type=int, default=10, help="history length")
    parser.add_argument("--test-history-len", type=int, default=20, help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1, help="dilate history graph")

    parser.add_argument("--grid-search", action='store_true', default=False, help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases", help="stat to use")
    parser.add_argument("--num-k", type=int, default=500, help="number of triples generated")

    # --- STRIDE curriculum learning args (new) ---
    parser.add_argument("--use-curriculum", action='store_true', default=True,
                         help="enable STRIDE curriculum learning (snapshot difficulty pacing)")
    parser.add_argument("--no-curriculum", dest="use_curriculum", action='store_false',
                         help="disable curriculum learning, fall back to uniform shuffled training")
    parser.add_argument("--curriculum-warmup-ratio", type=float, default=0.4,
                         help="fraction of n-epochs spent ramping up from start-ratio to full data")
    parser.add_argument("--curriculum-strategy", type=str, default="adaptive",
                         choices=["adaptive", "linear", "cosine"],
                         help="curriculum ratio-growth strategy")

    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, 'a')
            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr_raw))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
    else:
        run_experiment(args)
    sys.exit()

# # @Time    : 2019-08-10 11:20
# # @Author  : Lee_zix
# # @Email   : Lee_zix@163.com
# # @File    : main.py
# # @Software: PyCharm
# """
# The entry of the KGEvolve
# """

# import argparse
# import itertools
# import os
# import sys
# import time
# import pickle

# import dgl
# import numpy as np
# import torch
# from tqdm import tqdm
# import random
# sys.path.append("..")
# from rgcn import utils
# from rgcn.utils import build_sub_graph
# from src.rrgcn import RecurrentRGCN
# from src.hyperparameter_range import hp_range
# import torch.nn.modules.rnn
# from collections import defaultdict
# from rgcn.knowledge_graph import _read_triplets_as_list
# # os.environ['KMP_DUPLICATE_LIB_OK']='True'


# def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
#     """
#     :param model: model used to test
#     :param history_list:    all input history snap shot list, not include output label train list or valid list
#     :param test_list:   test triple snap shot list
#     :param num_rels:    number of relations
#     :param num_nodes:   number of nodes
#     :param use_cuda:
#     :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
#     :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
#     :param model_name:
#     :param static_graph
#     :param mode
#     :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
#     """
#     ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
#     ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

#     idx = 0
#     if mode == "test":
#         # test mode: load parameter form file
#         if use_cuda:
#             checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
#         else:
#             checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
#         print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
#         print("\n"+"-"*10+"start testing"+"-"*10+"\n")
#         model.load_state_dict(checkpoint['state_dict'])

#     model.eval()
#     # do not have inverse relation in test input
#     input_list = [snap for snap in history_list[-args.test_history_len:]]

#     for time_idx, test_snap in enumerate(tqdm(test_list)):
#         history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
#         test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
#         test_triples_input = test_triples_input.to(args.gpu)
#         test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda)

#         mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
#         mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

#         # used to global statistic
#         ranks_raw.append(rank_raw)
#         ranks_filter.append(rank_filter)
#         # used to show slide results
#         mrr_raw_list.append(mrr_snap)
#         mrr_filter_list.append(mrr_filter_snap)

#         # relation rank
#         ranks_raw_r.append(rank_raw_r)
#         ranks_filter_r.append(rank_filter_r)
#         mrr_raw_list_r.append(mrr_snap_r)
#         mrr_filter_list_r.append(mrr_filter_snap_r)

#         # reconstruct history graph list
#         if args.multi_step:
#             if not args.relation_evaluation:    
#                 predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
#             else:
#                 predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
#             if len(predicted_snap):
#                 input_list.pop(0)
#                 input_list.append(predicted_snap)
#         else:
#             input_list.pop(0)
#             input_list.append(test_snap)
#         idx += 1
    
#     mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
#     mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
#     mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
#     mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
#     return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


# def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
#     # load configuration for grid search the best configuration
#     if n_hidden:
#         args.n_hidden = n_hidden
#     if n_layers:
#         args.n_layers = n_layers
#     if dropout:
#         args.dropout = dropout
#     if n_bases:
#         args.n_bases = n_bases

#     # load graph data
#     print("loading graph data")
#     data = utils.load_data(args.dataset)
#     train_list = utils.split_by_time(data.train)
#     valid_list = utils.split_by_time(data.valid)
#     test_list = utils.split_by_time(data.test)

#     num_nodes = data.num_nodes
#     num_rels = data.num_rels

#     all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
#     all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
#     all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
#     all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

#     model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}"\
#         .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
#                 args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
#     model_state_file = '../models/' + model_name
#     print("Sanity Check: stat name : {}".format(model_state_file))
#     print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

#     use_cuda = args.gpu >= 0 and torch.cuda.is_available()

#     if args.add_static_graph:
#         static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
#         num_static_rels = len(np.unique(static_triples[:, 1]))
#         num_words = len(np.unique(static_triples[:, 2]))
#         static_triples[:, 2] = static_triples[:, 2] + num_nodes 
#         static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
#             if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
#     else:
#         num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

#     # create stat
#     model = RecurrentRGCN(args.decoder,
#                           args.encoder,
#                         num_nodes,
#                         num_rels,
#                         num_static_rels,
#                         num_words,
#                         args.n_hidden,
#                         args.opn,
#                         sequence_len=args.train_history_len,
#                         num_bases=args.n_bases,
#                         num_basis=args.n_basis,
#                         num_hidden_layers=args.n_layers,
#                         dropout=args.dropout,
#                         self_loop=args.self_loop,
#                         skip_connect=args.skip_connect,
#                         layer_norm=args.layer_norm,
#                         input_dropout=args.input_dropout,
#                         hidden_dropout=args.hidden_dropout,
#                         feat_dropout=args.feat_dropout,
#                         aggregation=args.aggregation,
#                         weight=args.weight,
#                         discount=args.discount,
#                         angle=args.angle,
#                         use_static=args.add_static_graph,
#                         entity_prediction=args.entity_prediction,
#                         relation_prediction=args.relation_prediction,
#                         use_cuda=use_cuda,
#                         gpu = args.gpu,
#                         analysis=args.run_analysis)

#     if use_cuda:
#         torch.cuda.set_device(args.gpu)
#         model.cuda()

#     if args.add_static_graph:
#         static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

#     if args.test and os.path.exists(model_state_file):
#         mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
#                                                             train_list+valid_list, 
#                                                             test_list, 
#                                                             num_rels, 
#                                                             num_nodes, 
#                                                             use_cuda, 
#                                                             all_ans_list_test, 
#                                                             all_ans_list_r_test, 
#                                                             model_state_file, 
#                                                             static_graph, 
#                                                             "test")
#     elif args.test and not os.path.exists(model_state_file):
#         print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
#     else:
#         print("----------------------------------------start training----------------------------------------\n")
#         best_mrr = 0
#         for epoch in range(args.n_epochs):
#             model.train()
#             losses = []
#             losses_e = []
#             losses_r = []
#             losses_static = []

#             idx = [_ for _ in range(len(train_list))]
#             random.shuffle(idx)

#             for train_sample_num in tqdm(idx):
#                 if train_sample_num == 0: continue
#                 output = train_list[train_sample_num:train_sample_num+1]
#                 if train_sample_num - args.train_history_len<0:
#                     input_list = train_list[0: train_sample_num]
#                 else:
#                     input_list = train_list[train_sample_num - args.train_history_len:
#                                         train_sample_num]

#                 # generate history graph
#                 history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
#                 output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
#                 loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda)
#                 loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

#                 losses.append(loss.item())
#                 losses_e.append(loss_e.item())
#                 losses_r.append(loss_r.item())
#                 losses_static.append(loss_static.item())

#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
#                 optimizer.step()
#                 optimizer.zero_grad()

#             print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
#                   .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

#             # validation
#             if epoch and epoch % args.evaluate_every == 0:
#                 mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
#                                                                     train_list, 
#                                                                     valid_list, 
#                                                                     num_rels, 
#                                                                     num_nodes, 
#                                                                     use_cuda, 
#                                                                     all_ans_list_valid, 
#                                                                     all_ans_list_r_valid, 
#                                                                     model_state_file, 
#                                                                     static_graph, 
#                                                                     mode="train")
                
#                 if not args.relation_evaluation:  # entity prediction evalution
#                     if mrr_raw < best_mrr:
#                         if epoch >= args.n_epochs:
#                             break
#                     else:
#                         best_mrr = mrr_raw
#                         torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
#                 else:
#                     if mrr_raw_r < best_mrr:
#                         if epoch >= args.n_epochs:
#                             break
#                     else:
#                         best_mrr = mrr_raw_r
#                         torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
#         mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
#                                                             train_list+valid_list,
#                                                             test_list, 
#                                                             num_rels, 
#                                                             num_nodes, 
#                                                             use_cuda, 
#                                                             all_ans_list_test, 
#                                                             all_ans_list_r_test, 
#                                                             model_state_file, 
#                                                             static_graph, 
#                                                             mode="test")
#     return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='REGCN')

#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--batch-size", type=int, default=1,
#                         help="batch-size")
#     parser.add_argument("-d", "--dataset", type=str, required=True,
#                         help="dataset to use")
#     parser.add_argument("--test", action='store_true', default=False,
#                         help="load stat from dir and directly test")
#     parser.add_argument("--run-analysis", action='store_true', default=False,
#                         help="print log info")
#     parser.add_argument("--run-statistic", action='store_true', default=False,
#                         help="statistic the result")
#     parser.add_argument("--multi-step", action='store_true', default=False,
#                         help="do multi-steps inference without ground truth")
#     parser.add_argument("--topk", type=int, default=10,
#                         help="choose top k entities as results when do multi-steps without ground truth")
#     parser.add_argument("--add-static-graph",  action='store_true', default=False,
#                         help="use the info of static graph")
#     parser.add_argument("--add-rel-word", action='store_true', default=False,
#                         help="use words in relaitons")
#     parser.add_argument("--relation-evaluation", action='store_true', default=False,
#                         help="save model accordding to the relation evalution")

#     # configuration for encoder RGCN stat
#     parser.add_argument("--weight", type=float, default=1,
#                         help="weight of static constraint")
#     parser.add_argument("--task-weight", type=float, default=0.7,
#                         help="weight of entity prediction task")
#     parser.add_argument("--discount", type=float, default=1,
#                         help="discount of weight of static constraint")
#     parser.add_argument("--angle", type=int, default=10,
#                         help="evolution speed")

#     parser.add_argument("--encoder", type=str, default="uvrgcn",
#                         help="method of encoder")
#     parser.add_argument("--aggregation", type=str, default="none",
#                         help="method of aggregation")
#     parser.add_argument("--dropout", type=float, default=0.2,
#                         help="dropout probability")
#     parser.add_argument("--skip-connect", action='store_true', default=False,
#                         help="whether to use skip connect in a RGCN Unit")
#     parser.add_argument("--n-hidden", type=int, default=200,
#                         help="number of hidden units")
#     parser.add_argument("--opn", type=str, default="sub",
#                         help="opn of compgcn")

#     parser.add_argument("--n-bases", type=int, default=100,
#                         help="number of weight blocks for each relation")
#     parser.add_argument("--n-basis", type=int, default=100,
#                         help="number of basis vector for compgcn")
#     parser.add_argument("--n-layers", type=int, default=2,
#                         help="number of propagation rounds")
#     parser.add_argument("--self-loop", action='store_true', default=True,
#                         help="perform layer normalization in every layer of gcn ")
#     parser.add_argument("--layer-norm", action='store_true', default=False,
#                         help="perform layer normalization in every layer of gcn ")
#     parser.add_argument("--relation-prediction", action='store_true', default=False,
#                         help="add relation prediction loss")
#     parser.add_argument("--entity-prediction", action='store_true', default=False,
#                         help="add entity prediction loss")
#     parser.add_argument("--split_by_relation", action='store_true', default=False,
#                         help="do relation prediction")

#     # configuration for stat training
#     parser.add_argument("--n-epochs", type=int, default=500,
#                         help="number of minimum training epochs on each time step")
#     parser.add_argument("--lr", type=float, default=0.001,
#                         help="learning rate")
#     parser.add_argument("--grad-norm", type=float, default=1.0,
#                         help="norm to clip gradient to")

#     # configuration for evaluating
#     parser.add_argument("--evaluate-every", type=int, default=20,
#                         help="perform evaluation every n epochs")

#     # configuration for decoder
#     parser.add_argument("--decoder", type=str, default="convtranse",
#                         help="method of decoder")
#     parser.add_argument("--input-dropout", type=float, default=0.2,
#                         help="input dropout for decoder ")
#     parser.add_argument("--hidden-dropout", type=float, default=0.2,
#                         help="hidden dropout for decoder")
#     parser.add_argument("--feat-dropout", type=float, default=0.2,
#                         help="feat dropout for decoder")

#     # configuration for sequences stat
#     parser.add_argument("--train-history-len", type=int, default=10,
#                         help="history length")
#     parser.add_argument("--test-history-len", type=int, default=20,
#                         help="history length for test")
#     parser.add_argument("--dilate-len", type=int, default=1,
#                         help="dilate history graph")

#     # configuration for optimal parameters
#     parser.add_argument("--grid-search", action='store_true', default=False,
#                         help="perform grid search for best configuration")
#     parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
#                         help="stat to use")
#     parser.add_argument("--num-k", type=int, default=500,
#                         help="number of triples generated")


#     args = parser.parse_args()
#     print(args)
#     if args.grid_search:
#         out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
#         o_f = open(out_log, 'w')
#         print("** Grid Search **")
#         o_f.write("** Grid Search **\n")
#         hyperparameters = args.tune.split(',')

#         if args.tune == '' or len(hyperparameters) < 1:
#             print("No hyperparameter specified.")
#             sys.exit(0)
#         grid = hp_range[hyperparameters[0]]
#         for hp in hyperparameters[1:]:
#             grid = itertools.product(grid, hp_range[hp])
#         hits_at_1s = {}
#         hits_at_10s = {}
#         mrrs = {}
#         grid = list(grid)
#         print('* {} hyperparameter combinations to try'.format(len(grid)))
#         o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
#         o_f.close()

#         for i, grid_entry in enumerate(list(grid)):

#             o_f = open(out_log, 'a')

#             if not (type(grid_entry) is list or type(grid_entry) is list):
#                 grid_entry = [grid_entry]
#             grid_entry = utils.flatten(grid_entry)
#             print('* Hyperparameter Set {}:'.format(i))
#             o_f.write('* Hyperparameter Set {}:\n'.format(i))
#             signature = ''
#             print(grid_entry)
#             o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
#             # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
#             mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
#             print("MRR (raw): {:.6f}".format(mrr))
#             o_f.write("MRR (raw): {:.6f}\n".format(mrr))
#             for hit in hits:
#                 avg_count = torch.mean((ranks <= hit).float())
#                 print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
#                 o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
#     # single run
#     else:
#         run_experiment(args)
#     sys.exit()


