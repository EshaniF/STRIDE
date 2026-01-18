import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def diagnose_difficulty_computation(train_list):
    """
    Diagnostic tool to understand why difficulty scores are compressed.
    Run this to see what's happening with your data.
    """
    print("="*60)
    print("DIFFICULTY SCORE DIAGNOSTIC REPORT")
    print("="*60)
    
    # Basic dataset statistics
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   Total snapshots: {len(train_list)}")
    print(f"   Snapshot sizes: min={min(len(s) for s in train_list)}, "
          f"max={max(len(s) for s in train_list)}, "
          f"mean={np.mean([len(s) for s in train_list]):.1f}")
    
    # Collect all triples
    all_triples = np.concatenate(train_list) if train_list else np.array([])
    
    if len(all_triples) == 0:
        print("   ERROR: No triples found!")
        return
    
    print(f"   Total triples: {len(all_triples)}")
    
    # Entity and relation analysis
    entity_freq = defaultdict(int)
    relation_freq = defaultdict(int)
    entity_degree = defaultdict(set)
    
    for triple in all_triples:
        s, r, o = triple
        entity_freq[s] += 1
        entity_freq[o] += 1
        relation_freq[r] += 1
        entity_degree[s].add(r)
        entity_degree[o].add(r)
    
    print(f"\n2. ENTITY STATISTICS:")
    entity_freqs = list(entity_freq.values())
    print(f"   Unique entities: {len(entity_freq)}")
    print(f"   Frequency: min={min(entity_freqs)}, max={max(entity_freqs)}, "
          f"mean={np.mean(entity_freqs):.1f}, std={np.std(entity_freqs):.1f}")
    print(f"   Frequency distribution:")
    print(f"     Quartiles: {np.percentile(entity_freqs, [25, 50, 75])}")
    print(f"     Top 5 most frequent: {sorted(entity_freqs, reverse=True)[:5]}")
    print(f"     Bottom 5 least frequent: {sorted(entity_freqs)[:5]}")
    
    print(f"\n3. RELATION STATISTICS:")
    relation_freqs = list(relation_freq.values())
    print(f"   Unique relations: {len(relation_freq)}")
    print(f"   Frequency: min={min(relation_freqs)}, max={max(relation_freqs)}, "
          f"mean={np.mean(relation_freqs):.1f}, std={np.std(relation_freqs):.1f}")
    
    print(f"\n4. ENTITY DEGREE STATISTICS:")
    degrees = [len(degree_set) for degree_set in entity_degree.values()]
    print(f"   Degree: min={min(degrees)}, max={max(degrees)}, "
          f"mean={np.mean(degrees):.1f}, std={np.std(degrees):.1f}")
    
    # Per-snapshot analysis
    print(f"\n5. PER-SNAPSHOT METRICS:")
    
    snapshot_stats = []
    for t, snap in enumerate(train_list):
        if len(snap) == 0:
            continue
        
        # Calculate metrics for this snapshot
        snap_entities = set()
        snap_relations = set()
        snap_entity_freqs = []
        snap_relation_freqs = []
        snap_degrees = []
        
        for triple in snap:
            s, r, o = triple
            snap_entities.add(s)
            snap_entities.add(o)
            snap_relations.add(r)
            snap_entity_freqs.append(entity_freq[s])
            snap_entity_freqs.append(entity_freq[o])
            snap_relation_freqs.append(relation_freq[r])
            snap_degrees.append(len(entity_degree[s]))
            snap_degrees.append(len(entity_degree[o]))
        
        stats = {
            'index': t,
            'size': len(snap),
            'unique_entities': len(snap_entities),
            'unique_relations': len(snap_relations),
            'avg_entity_freq': np.mean(snap_entity_freqs),
            'avg_relation_freq': np.mean(snap_relation_freqs),
            'avg_degree': np.mean(snap_degrees),
            'entity_diversity': len(snap_entities) / len(snap),
            'relation_diversity': len(snap_relations) / len(snap)
        }
        snapshot_stats.append(stats)
    
    # Show first, middle, and last snapshots
    print(f"\n   First snapshot (t=0):")
    if snapshot_stats:
        print(f"     {snapshot_stats[0]}")
    
    print(f"\n   Middle snapshot (t={len(snapshot_stats)//2}):")
    if len(snapshot_stats) > 1:
        print(f"     {snapshot_stats[len(snapshot_stats)//2]}")
    
    print(f"\n   Last snapshot (t={len(snapshot_stats)-1}):")
    if snapshot_stats:
        print(f"     {snapshot_stats[-1]}")
    
    # Check for problematic patterns
    print(f"\n6. POTENTIAL ISSUES:")
    
    # Issue 1: Very similar entity frequencies across snapshots
    avg_freqs = [s['avg_entity_freq'] for s in snapshot_stats]
    freq_variance = np.std(avg_freqs) / np.mean(avg_freqs) if np.mean(avg_freqs) > 0 else 0
    print(f"   Entity frequency variance across snapshots: {freq_variance:.4f}")
    if freq_variance < 0.1:
        print(f"   ⚠️  WARNING: Very low variance in entity frequencies!")
        print(f"       All snapshots might have similar entity frequency profiles.")
    
    # Issue 2: Similar snapshot sizes
    sizes = [s['size'] for s in snapshot_stats]
    size_variance = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    print(f"   Snapshot size variance: {size_variance:.4f}")
    if size_variance < 0.1:
        print(f"   ⚠️  WARNING: Very uniform snapshot sizes!")
    
    # Issue 3: Similar degree distributions
    avg_degrees = [s['avg_degree'] for s in snapshot_stats]
    degree_variance = np.std(avg_degrees) / np.mean(avg_degrees) if np.mean(avg_degrees) > 0 else 0
    print(f"   Degree variance across snapshots: {degree_variance:.4f}")
    if degree_variance < 0.1:
        print(f"   ⚠️  WARNING: Very uniform degree distributions!")
    
    # Visualization
    print(f"\n7. CREATING VISUALIZATION...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Snapshot sizes
    axes[0, 0].plot([s['size'] for s in snapshot_stats], 'b-o')
    axes[0, 0].set_title('Snapshot Sizes Over Time')
    axes[0, 0].set_xlabel('Snapshot Index')
    axes[0, 0].set_ylabel('Number of Triples')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average entity frequency
    axes[0, 1].plot(avg_freqs, 'r-o')
    axes[0, 1].set_title('Avg Entity Frequency Over Time')
    axes[0, 1].set_xlabel('Snapshot Index')
    axes[0, 1].set_ylabel('Average Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average degree
    axes[0, 2].plot(avg_degrees, 'g-o')
    axes[0, 2].set_title('Avg Entity Degree Over Time')
    axes[0, 2].set_xlabel('Snapshot Index')
    axes[0, 2].set_ylabel('Average Degree')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Entity frequency distribution
    axes[1, 0].hist(entity_freqs, bins=50, edgecolor='black')
    axes[1, 0].set_title('Entity Frequency Distribution')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Relation frequency distribution
    axes[1, 1].hist(relation_freqs, bins=min(50, len(relation_freqs)), edgecolor='black')
    axes[1, 1].set_title('Relation Frequency Distribution')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Degree distribution
    axes[1, 2].hist(degrees, bins=50, edgecolor='black')
    axes[1, 2].set_title('Entity Degree Distribution')
    axes[1, 2].set_xlabel('Degree')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('difficulty_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to 'difficulty_diagnostic.png'")
    
    print(f"\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    return snapshot_stats