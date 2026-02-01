import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os

class AdaptiveWeightPrinter:
    """
    Simplified tracker that just prints the adaptive weight values
    since they remain constant throughout training.
    """
    
    def print_adaptive_weights(self, difficulty_analyzer, ablation_mode='all', dataset_name=''):
        """
        Print the adaptive weight values calculated from dataset statistics.
        
        Args:
            difficulty_analyzer: GeneralDifficultyAnalyzer instance
            ablation_mode: Which ablation mode is being used
            dataset_name: Name of the dataset
        """
        print("\n" + "="*60)
        print(f"ADAPTIVE WEIGHT VALUES - {dataset_name}")
        print("="*60)
        
        # Calculate weights based on ablation mode (same logic as in your code)
        if ablation_mode == 'first_2' or ablation_mode == 'first_3' or ablation_mode == 'all':
            # Calculate frequency weight
            freq_weight = min(0.6, 0.3 + difficulty_analyzer.entity_freq_std / 
                            max(difficulty_analyzer.entity_freq_mean, 1) * 0.1)
            
            # Calculate temporal weight based on ablation mode
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
            temporal_weight = 1.0
            freq_weight = 0.0
            degree_weight = 0.0
            size_weight = 0.0
        else:  # 'none'
            temporal_weight = 0.5
            freq_weight = 0.0
            degree_weight = 0.0
            size_weight = 0.0
        
        # Print weight values
        print(f"\nAblation Mode: {ablation_mode}")
        print(f"\nDataset Statistics:")
        print(f"  Entity Frequency Mean: {difficulty_analyzer.entity_freq_mean:.4f}")
        print(f"  Entity Frequency Std:  {difficulty_analyzer.entity_freq_std:.4f}")
        print(f"  Std/Mean Ratio:        {difficulty_analyzer.entity_freq_std / max(difficulty_analyzer.entity_freq_mean, 1):.4f}")
        
        print(f"\nCalculated Adaptive Weights:")
        print(f"  Temporal Weight:  {temporal_weight:.4f}")
        print(f"  Frequency Weight: {freq_weight:.4f}")
        print(f"  Degree Weight:    {degree_weight:.4f}")
        print(f"  Size Weight:      {size_weight:.4f}")
        print(f"  Total:            {temporal_weight + freq_weight + degree_weight + size_weight:.4f}")
        
        print(f"\nInterpretation:")
        if freq_weight > temporal_weight:
            print(f"  → Frequency-focused: {freq_weight:.1%} weight on structural patterns")
            print(f"  → Higher entity frequency variance detected in dataset")
        elif temporal_weight > freq_weight:
            print(f"  → Temporal-focused: {temporal_weight:.1%} weight on temporal patterns")
            print(f"  → Lower entity frequency variance detected in dataset")
        else:
            print(f"  → Balanced: Equal weight on temporal and frequency patterns")
        
        print("\nNote: These weights remain CONSTANT throughout training.")
        print("They are calculated once from dataset statistics, not learned.")
        print("="*60 + "\n")
        
        return {
            'temporal_weight': temporal_weight,
            'freq_weight': freq_weight,
            'degree_weight': degree_weight,
            'size_weight': size_weight
        }

