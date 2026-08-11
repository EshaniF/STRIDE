#Training script for curriculum learning
import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import json
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
from scipy import stats
# from adaptive_weight_test import AdaptiveWeightTracker
from adaptive_weight_test import AdaptiveWeightPrinter

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
            
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (
                0.5 * (1 - np.cos(progress * np.pi)) + 0.3 * progress
            )
        elif self.strategy == 'cosine':
            # Cosine progression
            ratio = self.start_ratio + (1.0 - self.start_ratio) * (1 - np.cos(progress * np.pi)) / 2
        else:
            # Linear progression 
            ratio = self.start_ratio + (1.0 - self.start_ratio) * progress
            
        return min(ratio, 1.0)

class GeneralDifficultyAnalyzer:
    """
    General difficulty analyzer that works for any TKG dataset.
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
        Compute difficulty scores using multiple metrics.
        Ensures all scores are normalized to [0, 1] range.
        """
        difficulty_scores = []
        
        # Pre-compute normalization factors - collect SNAPSHOT-LEVEL averages
        all_frequency_scores = []  # per-snapshot averages
        all_degree_scores = []      
        all_size_scores = []        
        
        # collect raw SNAPSHOT scores for normalization
        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                continue
                
            # Collect frequency scores (average per snapshot)
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
            
            # Collect degree scores (average per snapshot)
            if self.ablation_mode in ['all', 'first_3']:
                snap_degree_scores = []
                for triple in snap:
                    s, r, o = triple
                    s_degree = len(self.entity_degree.get(s, set()))
                    o_degree = len(self.entity_degree.get(o, set()))
                    snap_degree_scores.append((s_degree + o_degree) / 2.0)
                
                if snap_degree_scores:
                    all_degree_scores.append(np.mean(snap_degree_scores))
            
            # Collect size scores
            if self.ablation_mode == 'all':
                all_size_scores.append(len(snap))
        
        # Compute normalization parameters from SNAPSHOT-LEVEL scores
        freq_min = min(all_frequency_scores) if all_frequency_scores else 0
        freq_max = max(all_frequency_scores) if all_frequency_scores else 1
        freq_range = max(freq_max - freq_min, 1e-8)  # Avoid division by zero
        
        degree_min = min(all_degree_scores) if all_degree_scores else 0
        degree_max = max(all_degree_scores) if all_degree_scores else 1
        degree_range = max(degree_max - degree_min, 1e-8)
        
        size_min = min(all_size_scores) if all_size_scores else 0
        size_max = max(all_size_scores) if all_size_scores else 1
        size_range = max(size_max - size_min, 1e-8)
        
        # compute normalized difficulty scores
        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                difficulty_scores.append(0.5)
                continue
            
            # uniform difficulty
            if self.ablation_mode == 'none':
                difficulty_scores.append(0.5)
                continue
            
            # Temporal difficulty 
            temporal_score = t / max(len(self.train_list) - 1, 1)
            
            # Normalized frequency-based difficulty
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
                    
                    # Normalize to [0, 1]
                    normalized_score = (raw_score - freq_min) / freq_range
                    frequency_scores_snap.append(normalized_score)
                
                frequency_score = np.mean(frequency_scores_snap)
            
            # Normalized structural complexity (entity degree diversity)
            degree_score = 0.0
            if self.ablation_mode in ['all', 'first_3']:
                degree_scores_snap = []
                for triple in snap:
                    s, r, o = triple
                    s_degree = len(self.entity_degree.get(s, set()))
                    o_degree = len(self.entity_degree.get(o, set()))
                    raw_degree = (s_degree + o_degree) / 2.0
                    
                    # Normalize to [0, 1]
                    normalized_degree = (raw_degree - degree_min) / degree_range
                    degree_scores_snap.append(normalized_degree)
                
                degree_score = np.mean(degree_scores_snap) if degree_scores_snap else 0
            
            # Normalized snapshot size complexity
            size_score = 0.0
            if self.ablation_mode == 'all':
                size_score = (len(snap) - size_min) / size_range
            
            # Combine difficulty metrics with adaptive weights
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
                freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.05)
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
        
        # normalization
        if len(scores_array) > 0:
            scores_min = scores_array.min()
            scores_max = scores_array.max()
            if scores_max > scores_min:
                # Re-normalize to ensure full [0, 1] range utilization
                scores_array = (scores_array - scores_min) / (scores_max - scores_min)
        
        return scores_array

    # def compute_difficulty_scores(self):
    #     """
    #     Compute difficulty scores using multiple metrics.
    #     ablation mode.
    #     """
    #     difficulty_scores = []
        
    #     for t, snap in enumerate(self.train_list):
    #         if len(snap) == 0:
    #             difficulty_scores.append(0.5)  # Neutral difficulty for empty snapshots
    #             continue
            
    #         # Baseline: uniform difficulty
    #         if self.ablation_mode == 'none':
    #             difficulty_scores.append(0.5)
    #             continue
            
    #         # Temporal difficulty 
    #         temporal_score = t / len(self.train_list)
            
    #         # Frequency-based difficulty (rare entities/relations = harder)
    #         frequency_score = 0.0
    #         if self.ablation_mode in ['all', 'first_3', 'first_2']:
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
            
    #         # Structural complexity (entity degree diversity)
    #         degree_score = 0.0
    #         if self.ablation_mode in ['all', 'first_3']:
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
            
    #         # Snapshot size complexity (larger snapshots = potentially harder)
    #         size_score = 0.0
    #         if self.ablation_mode == 'all':
    #             size_score = len(snap) / max(len(s) for s in self.train_list)
            
    #         # Combine difficulty metrics with adaptive weights based on ablation mode
    #         if self.ablation_mode == 'first_1':
    #             # Only temporal
    #             combined_score = temporal_score
                
    #         elif self.ablation_mode == 'first_2':
    #             # Temporal + Frequency
    #             freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
    #             temporal_weight = 1.0 - freq_weight
                
    #             combined_score = (
    #                 temporal_weight * temporal_score +
    #                 freq_weight * frequency_score
    #             )
                
    #         elif self.ablation_mode == 'first_3':
    #             # Temporal + Frequency + Degree
    #             freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
    #             temporal_weight = 0.85 - freq_weight
                
    #             combined_score = (
    #                 temporal_weight * temporal_score +
    #                 freq_weight * frequency_score +
    #                 0.15 * degree_score
    #             )
                
    #         else:  # 'all'
    #             # All 4 components
    #             freq_weight = min(0.6, 0.3 + self.entity_freq_std / max(self.entity_freq_mean, 1) * 0.1)
    #             temporal_weight = 0.8 - freq_weight
                
    #             combined_score = (
    #                 temporal_weight * temporal_score +
    #                 freq_weight * frequency_score +
    #                 0.15 * degree_score +
    #                 0.05 * size_score
    #             )
            
    #         difficulty_scores.append(combined_score)
        
    #     return np.array(difficulty_scores)


def get_curriculum_samples(train_list, difficulty_scores, current_ratio):
    """
    Get training samples.
    """
    n_samples = max(1, min(int(len(train_list) * current_ratio), len(train_list)))
    
    
    # Create temporal preference (earlier samples preferred)
    temporal_preference = np.arange(len(train_list)) / len(train_list)
    
    # Combine temporal and difficulty scores

    temporal_weight = max(0.3, 1.0 - current_ratio)  # Decrease temporal weight as curriculum progresses
    difficulty_weight = 1.0 - temporal_weight
    
    combined_scores = (
        temporal_weight * temporal_preference +
        difficulty_weight * difficulty_scores
    )
    
    # # Select samples with combined scoring
    # selected_indices = np.argsort(combined_scores)[:n_samples]
    
    # # Ensure to include some early temporal samples for context
    # min_temporal_samples = max(1, min(n_samples // 4, 5))
    # early_samples = list(range(1, min(min_temporal_samples + 1, len(train_list))))
    
    # # Combine and deduplicate
    # selected_indices = sorted(list(set(selected_indices) | set(early_samples)))
    
    # return selected_indices[:n_samples]  # Ensure don't exceed target sample count
    # Select samples with combined scoring
    ranked_indices = np.argsort(combined_scores)  # easiest -> hardest, full ranking
    selected_set = set(ranked_indices[:n_samples].tolist())

    # Ensure to include some early temporal samples for context
    min_temporal_samples = max(1, min(n_samples // 4, 5))
    early_samples = set(range(1, min(min_temporal_samples + 1, len(train_list))))

    # Union in the forced early samples, then re-rank the UNION by
    # combined_scores (not by chronological index) before truncating back
    # to n_samples -- sorting by index and slicing here was the bug: it
    # silently discarded the difficulty ranking once the union pushed the
    # set above n_samples, degrading the curriculum toward "train on the
    # earliest snapshots" rather than "train on the n_samples easiest
    # snapshots by combined score."
    union_indices = selected_set | early_samples
    union_sorted_by_difficulty = sorted(union_indices, key=lambda idx: combined_scores[idx])

    return union_sorted_by_difficulty[:n_samples]

class DifficultyVisualizer:
    """
    Visualizer for temporal knowledge graph difficulty landscape.
    Creates histograms, KDE plots, and correlation heatmaps for difficulty metrics.
    """
    
    def __init__(self, difficulty_analyzer, train_list):
        """
        Args:
            difficulty_analyzer: Instance of GeneralDifficultyAnalyzer
            train_list: List of training snapshots
        """
        self.analyzer = difficulty_analyzer
        self.train_list = train_list
        self.difficulty_scores = None
        self.component_scores = None
        
    def compute_component_scores(self):
        """
        Compute individual difficulty components for each snapshot.
        Returns a dictionary with temporal, frequency, degree, and size scores.
        All scores are normalized to [0, 1] range.
        """
        temporal_scores = []
        frequency_scores = []
        degree_scores = []
        size_scores = []
        
        # First pass: collect raw snapshot-level scores for normalization
        raw_frequency_scores = []
        raw_degree_scores = []
        raw_size_scores = []
        
        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                continue
            
            # Collect frequency scores (average per snapshot)
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
            
            # Collect degree scores (average per snapshot)
            deg_vals = []
            for triple in snap:
                s, r, o = triple
                s_degree = len(self.analyzer.entity_degree.get(s, set()))
                o_degree = len(self.analyzer.entity_degree.get(o, set()))
                deg_vals.append((s_degree + o_degree) / 2.0)
            
            if deg_vals:
                raw_degree_scores.append(np.mean(deg_vals))
            
            # Collect size scores
            raw_size_scores.append(len(snap))
        
        # Compute normalization parameters from snapshot-level scores
        freq_min = min(raw_frequency_scores) if raw_frequency_scores else 0
        freq_max = max(raw_frequency_scores) if raw_frequency_scores else 1
        freq_range = max(freq_max - freq_min, 1e-8)
        
        degree_min = min(raw_degree_scores) if raw_degree_scores else 0
        degree_max = max(raw_degree_scores) if raw_degree_scores else 1
        degree_range = max(degree_max - degree_min, 1e-8)
        
        size_min = min(raw_size_scores) if raw_size_scores else 0
        size_max = max(raw_size_scores) if raw_size_scores else 1
        size_range = max(size_max - size_min, 1e-8)
        
        # Second pass: compute normalized scores
        for t, snap in enumerate(self.train_list):
            if len(snap) == 0:
                temporal_scores.append(0.5)
                frequency_scores.append(0.5)
                degree_scores.append(0.5)
                size_scores.append(0.5)
                continue
            
            # Temporal difficulty (already in [0, 1])
            temporal_score = t / max(len(self.train_list) - 1, 1)
            temporal_scores.append(temporal_score)
            
            # Normalized frequency-based difficulty
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
            normalized_freq_score = (raw_freq_score - freq_min) / freq_range
            frequency_scores.append(normalized_freq_score)
            
            # Normalized structural complexity (degree)
            deg_vals = []
            for triple in snap:
                s, r, o = triple
                s_degree = len(self.analyzer.entity_degree.get(s, set()))
                o_degree = len(self.analyzer.entity_degree.get(o, set()))
                deg_vals.append((s_degree + o_degree) / 2.0)
            
            raw_degree_score = np.mean(deg_vals) if deg_vals else 0
            normalized_degree_score = (raw_degree_score - degree_min) / degree_range
            degree_scores.append(normalized_degree_score)
            
            # Normalized snapshot size complexity
            raw_size_score = len(snap)
            normalized_size_score = (raw_size_score - size_min) / size_range
            size_scores.append(normalized_size_score)
        
        # Store normalized component scores
        self.component_scores = {
            'Temp': np.array(temporal_scores),
            'Freq': np.array(frequency_scores),
            'Struct': np.array(degree_scores),
            'Size': np.array(size_scores)
        }
        
        # Print verification statistics
        print("\nComponent Score Ranges (After Normalization):")
        for component, scores in self.component_scores.items():
            non_empty = scores[scores != 0.5]  # Exclude empty snapshots
            if len(non_empty) > 0:
                print(f"  {component}: [{non_empty.min():.4f}, {non_empty.max():.4f}] "
                    f"(mean={non_empty.mean():.4f}, std={non_empty.std():.4f})")
        
        return self.component_scores
    
    def plot_difficulty_distributions(self, save_path='difficulty_distributions.png', figsize=(16, 10)):
        """
        Create histogram and KDE plots for each difficulty component.
        """
        if self.component_scores is None:
            self.compute_component_scores()
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        component_names = ['Temp', 'Freq', 'Struct', 'Size']
        
        for idx, (name, color) in enumerate(zip(component_names, colors)):
            scores = self.component_scores[name]
            
            # Histogram
            ax_hist = axes[0, idx]
            ax_hist.hist(scores, bins=30, color=color, alpha=0.7, edgecolor='black', density=True)
            ax_hist.set_xlabel('Difficulty Score', fontsize=11, fontweight='bold')
            ax_hist.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax_hist.set_title(f'{name} Difficulty Distribution', fontsize=12, fontweight='bold')
            ax_hist.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax_hist.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Std: {std_val:.3f}')
            ax_hist.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax_hist.legend(fontsize=9)
            
            # KDE plot
            ax_kde = axes[1, idx]
            try:
                kde = stats.gaussian_kde(scores)
                x_range = np.linspace(scores.min(), scores.max(), 200)
                ax_kde.plot(x_range, kde(x_range), color=color, linewidth=2.5)
                ax_kde.fill_between(x_range, kde(x_range), alpha=0.3, color=color)
            except:
                # Fallback if KDE fails (e.g., all values the same)
                ax_kde.hist(scores, bins=20, color=color, alpha=0.5, density=True)
            
            ax_kde.set_xlabel('Difficulty Score', fontsize=11, fontweight='bold')
            ax_kde.set_ylabel('Density (KDE)', fontsize=11, fontweight='bold')
            ax_kde.set_title(f'{name} KDE', fontsize=12, fontweight='bold')
            ax_kde.grid(True, alpha=0.3, linestyle='--')
            
            # Add quartile lines
            q25, q50, q75 = np.percentile(scores, [25, 50, 75])
            ax_kde.axvline(q50, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Median: {q50:.3f}')
            ax_kde.axvline(q25, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Q1: {q25:.3f}')
            ax_kde.axvline(q75, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Q3: {q75:.3f}')
            ax_kde.legend(fontsize=8)
        
        plt.suptitle('Difficulty Component Distributions Across Training Snapshots', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved difficulty distributions to {save_path}")
        plt.close()
        
        return fig
    
    def plot_correlation_matrix(self, save_path='difficulty_correlation.png', figsize=(10, 8)):
        """
        Create correlation heatmap showing relationships between difficulty components.
        """
        if self.component_scores is None:
            self.compute_component_scores()
        
        # Create correlation matrix
        component_names = ['Temp', 'Freq', 'Struct', 'Size']
        n_components = len(component_names)
        corr_matrix = np.zeros((n_components, n_components))
        
        for i, name1 in enumerate(component_names):
            for j, name2 in enumerate(component_names):
                scores1 = self.component_scores[name1]
                scores2 = self.component_scores[name2]
                corr_matrix[i, j] = np.corrcoef(scores1, scores2)[0, 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_components))
        ax.set_yticks(np.arange(n_components))
        ax.set_xticklabels(component_names, fontsize=11, fontweight='bold')
        ax.set_yticklabels(component_names, fontsize=11, fontweight='bold')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values as text
        for i in range(n_components):
            for j in range(n_components):
                text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color=text_color,
                             fontsize=12, fontweight='bold')
        
        ax.set_title('Difficulty Component Correlation Matrix\n(Pearson Correlation)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation matrix to {save_path}")
        plt.close()
        
        # Print correlation analysis
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        for i, name1 in enumerate(component_names):
            for j, name2 in enumerate(component_names):
                if i < j:  # Only print upper triangle
                    corr = corr_matrix[i, j]
                    interpretation = self._interpret_correlation(corr)
                    print(f"{name1} vs {name2}: {corr:.3f} - {interpretation}")
        print("="*60 + "\n")
        
        return corr_matrix
    
    def _interpret_correlation(self, corr):
        """Interpret correlation coefficient magnitude."""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            strength = "Very Strong"
        elif abs_corr >= 0.6:
            strength = "Strong"
        elif abs_corr >= 0.4:
            strength = "Moderate"
        elif abs_corr >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "Positive" if corr >= 0 else "Negative"
        return f"{strength} {direction}"
    
    def plot_component_evolution(self, save_path='difficulty_evolution.png', figsize=(14, 8)):
        """
        Plot how each difficulty component evolves over time (snapshots).
        """
        if self.component_scores is None:
            self.compute_component_scores()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        component_names = ['Temp', 'Freq', 'Struct', 'Size']
        
        snapshot_indices = np.arange(len(self.train_list))
        
        for idx, (name, color) in enumerate(zip(component_names, colors)):
            ax = axes[idx]
            scores = self.component_scores[name]
            
            # Plot line
            ax.plot(snapshot_indices, scores, color=color, linewidth=2.5, 
                   marker='o', markersize=4, alpha=0.8, label=name)
            
            # Add trend line
            z = np.polyfit(snapshot_indices, scores, 2)  # Quadratic fit
            p = np.poly1d(z)
            ax.plot(snapshot_indices, p(snapshot_indices), 
                   color='red', linestyle='--', linewidth=2, alpha=0.6, label='Trend')
            
            ax.set_xlabel('Snapshot Index', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{name} Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{name} Difficulty Evolution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=9)
            
            # Fill between for visual effect
            ax.fill_between(snapshot_indices, scores, alpha=0.2, color=color)
        
        plt.suptitle('Difficulty Component Evolution Over Training Snapshots', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved difficulty evolution to {save_path}")
        plt.close()
        
        return fig
    
    def plot_combined_difficulty(self, save_path='combined_difficulty.png', figsize=(14, 6), dataset_name='ICEWS'):
        """
        Plot the final combined difficulty score compared to individual components.
        """
        if self.component_scores is None:
            self.compute_component_scores()
        
        # Compute combined difficulty using same weights as the analyzer
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
            # Fallback to equal weights
            combined = np.mean([self.component_scores[name] for name in ['Temp', 'Freq', 'Struct', 'Size']], axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        snapshot_indices = np.arange(len(self.train_list))
        
        # Plot 1: All components together
        # colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        colors = ["#1F77B4", "#C44E52", "#FFA600", "#2F9E44", "#B22222"]
        component_names = ['Temp', 'Freq', 'Struct', 'Size']
        
        for name, color in zip(component_names, colors):
            ax1.plot(snapshot_indices, self.component_scores[name], 
                    color=color, linewidth=3, alpha=0.7, label=name)
        
        ax1.plot(snapshot_indices, combined, color='black', linewidth=3, 
                linestyle='--', label='Composite', alpha=0.9)
        ax1.set_xlabel('Snapshot Index', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Difficulty Score', fontsize=18, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=18)
        # ax1.set_title(dataset_name, fontsize=18, fontweight='bold')
        ax1.legend(prop={'size': 18, 'weight': 'bold'}, loc=2, ncol=3, mode="expand", borderaxespad=0., framealpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Combined difficulty distribution
        ax2.hist(combined, bins=25, color='#4C72B0', alpha=0.7, edgecolor='black', density=True)
        kde = stats.gaussian_kde(combined)
        x_range = np.linspace(combined.min(), combined.max(), 200)
        ax2.plot(x_range, kde(x_range), color='#2B3A4A', linewidth=3, label='KDE')
        
        # Add KDE
        # try:
        #     kde = stats.gaussian_kde(combined)
        #     x_range = np.linspace(combined.min(), combined.max(), 200)
        #     ax2.plot(x_range, kde(x_range), color='darkviolet', linewidth=2.5, label='KDE')
        # except:
        #     pass
        
        mean_val = np.mean(combined)
        ax2.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=4, 
                   label=f'Mean: {mean_val:.2f}')
        
        ax2.set_xlabel('Composite Difficulty Score', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=18, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=18)
        # ax2.set_title('Combined Difficulty Distribution', fontsize=18, fontweight='bold')
        ax2.legend(prop={'size': 18, 'weight': 'bold'})
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Composite Difficulty Score Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined difficulty to {save_path}")
        plt.close()
        
        return combined
    
    def generate_all_visualizations(self, output_dir='./visualizations/', dataset_name='ICEWS'):
        """
        Generate all difficulty landscape visualizations at once.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING DIFFICULTY LANDSCAPE VISUALIZATIONS")
        print("="*60 + "\n")
        
        # 1. Distribution plots
        self.plot_difficulty_distributions(
            save_path=os.path.join(output_dir, 'difficulty_distributions.png')
        )
        
        # 2. Correlation matrix
        self.plot_correlation_matrix(
            save_path=os.path.join(output_dir, 'difficulty_correlation.png')
        )
        
        # 3. Evolution over time
        self.plot_component_evolution(
            save_path=os.path.join(output_dir, 'difficulty_evolution.png')
        )
        
        # 4. Combined difficulty
        self.plot_combined_difficulty(
            save_path=os.path.join(output_dir, 'combined_difficulty.png'), dataset_name='icews'
        )
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print(f"Output directory: {output_dir}")
        print("="*60 + "\n")
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _print_summary_statistics(self):
        """Print summary statistics for all difficulty components."""
        if self.component_scores is None:
            self.compute_component_scores()
        
        print("\n" + "="*60)
        print("DIFFICULTY COMPONENT STATISTICS")
        print("="*60)
        
        for name in ['Temp', 'Freq', 'Struct', 'Size']:
            scores = self.component_scores[name]
            print(f"\n{name}:")
            print(f"  Mean:   {np.mean(scores):.4f}")
            print(f"  Std:    {np.std(scores):.4f}")
            print(f"  Min:    {np.min(scores):.4f}")
            print(f"  Max:    {np.max(scores):.4f}")
            print(f"  Median: {np.median(scores):.4f}")
            print(f"  Range:  {np.ptp(scores):.4f}")
        
        print("="*60 + "\n")

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
    dst_set = set(triples[:, 2])

    # ----------------Second-order neighbor sampling-----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    
    # Integrate repeated triples and count the frequency of triples, using the frequency of triples as the fourth column of data
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #Convert query field to pandas

    dst_df = df_dic.query('sr in @er_list')  #Get query entities and relationships pandas
    dst_get = dst_df['dst'].values    #Get the target tail entity
    two_ent = set().union(*dst_get)   #Integrate head and tail entities
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  #Get query entities and relationships pandas
    dst_get_inv = dst_df_inv['dst'].values    #Get the target tail entity
    two_ent_inv = set().union(*dst_get_inv)   #Integrate head and tail entities
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')
    #----------------Second-order neighbor sampling-----------------------
    # result = subg_df.query('src in @src_set')
    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu) 
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    ranks_raw_r_inv, ranks_filter_r_inv, mrr_raw_list_r_inv, mrr_filter_list_r_inv = [], [], [], []
    ranks_raw1, ranks_filter1 = [],[]

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    # sr_to_sro = np.load('../data/{}/his_dict_new/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()
    sr_to_sro = np.load('../data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()

    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples =test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        
        que_pair =  e2r(test_snap, num_rels)
        
        que_pair_inv =  e2r(inverse_triples, num_rels)

        sub_snap,sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap , num_nodes,num_rels,use_cuda, args.gpu)

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
        test_triples, final_score = model.predict(que_pair, sub_snap, time_idx, history_glist, num_rels, static_graph, test_triples_input, use_cuda)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
            # used to show slide results
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            # else:
            #     predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
            # subg_arr = np.concatenate([subg_arr,test_snap])
            # print(np.shape(subg_arr))
        idx += 1

    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)
    print("(all_raw) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))
    
    #file dump
    if mode == "test": 
        filename = '../result/'+ args.dataset + ".csv"
        if os.path.isfile(filename) == False:# If the file does not exist, create it
            with open (filename,'w', newline='') as f:
                # Write column names
                fieldnames=['encoder','opn','pre_type','use_static','use_cl','gpu','datetime','pre_weight',
                            'train_len','test_len','temperature','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        #  data input
        with open (filename,'a', newline='') as f:
            writer = csv.writer(f)
            row={'encoder':args.encoder,'opn':args.opn,'pre_type':args.pre_type,'use_static':args.add_static_graph,'use_cl':args.use_cl,'gpu':args.gpu,'datetime':datetime.now(),'pre_weight':args.pre_weight,
                'train_len':args.train_history_len,'test_len':args.test_history_len,'temperature':args.temperature,'lr':args.lr,'n_hidden':args.n_hidden,
                'filter_MRR':float(mrr_filter),'filter_H@1':hit_filter[0],'filter_H@3':hit_filter[1],'filter_H@10':hit_filter[2],
                'filter_inv_MRR':float(mrr_filter_inv),'filter_inv_H@1':hit_filter_inv[0],'filter_inv_H@3':hit_filter_inv[1],'filter_inv_H@10':hit_filter_inv[2],
                'all_MRR':all_mrr_raw.item(),'all_H@1':all_hit_raw[0],'all_H@3':all_hit_raw[1],'all_H@10':all_hit_raw[2],
                'filter_all_MRR':all_mrr_filter.item(),'filter_all_H@1':all_hit_filter[0],'filter_all_H@3':all_hit_filter[1],'filter_all_H@10':all_hit_filter[2]}
            writer.writerow(row.values())
            
    return all_mrr_raw, all_mrr_filter

def visualize_difficulty_landscape(train_list, dataset_name='ICEWS'):
    """
    Standalone function to generate difficulty visualizations.
    
    Args:
        train_list: List of training snapshots from utils.split_by_time(data.train)
        dataset_name: Name of the dataset (for output directory)
    """
    from collections import defaultdict
    
    # Create difficulty analyzer
    difficulty_analyzer = GeneralDifficultyAnalyzer(train_list, ablation_mode='all')
    
    # Create visualizer
    visualizer = DifficultyVisualizer(difficulty_analyzer, train_list)
    
    # Generate all visualizations
    output_dir = f'./visualizations_{dataset_name}/'
    visualizer.generate_all_visualizations(output_dir=output_dir, dataset_name=dataset_name)
    
    return visualizer   

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

    print("Generating difficulty landscape visualizations...")
    visualizer = visualize_difficulty_landscape(train_list, args.dataset)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # Initialize general curriculum learning components
    use_curriculum = getattr(args, 'use_curriculum', True)
    
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
    weight_printer = AdaptiveWeightPrinter()
    if use_curriculum:
        weight_printer.print_adaptive_weights(
            difficulty_analyzer=difficulty_analyzer,
            ablation_mode='all',
            dataset_name=args.dataset.upper()
        )
    # Load answer lists for evaluation
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature, args.pre_weight, 
                args.use_cl, args.pre_type, args.n_hidden, args.encoder, str(time.time()))
    model_state_file = '../models/' + 'stride' 
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
                                    mode="train")
                # if use_curriculum:
                #     weight_tracker.record_weights(
                #         epoch=epoch,
                #         difficulty_analyzer=difficulty_analyzer,
                #         curriculum_scheduler=curriculum_scheduler,
                #         ablation_mode='all'
                #     )
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
                            mode="test")
                            
        print("="*60)
        print("Training completed successfully!")
        print("="*60)
        print('date time now is',datetime.now())
        
    return mrr_raw, mrr_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kgta')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="ICEWS18",
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
    parser.add_argument("--add-static-graph",  action='store_true', default=True,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="all",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=True,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.03,
                        help="the temperature of cl")
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.9,
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
    parser.add_argument("--n-epochs", type=int, default=70,
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
    parser.add_argument("--train-history-len", type=int, default=7,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=7,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")


    args = parser.parse_args()
    print(args)
    args.__dict__["test_history_len"] = args.__dict__["train_history_len"]

    run_experiment(args)