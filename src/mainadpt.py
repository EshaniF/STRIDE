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
# from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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

class DifficultyAnalyzer:
    """
    Analyzes training samples to determine their difficulty based on various metrics.
    """
    
    def __init__(self, train_list, num_nodes, num_rels):
        self.train_list = train_list
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.difficulty_scores = {}
        self._compute_statistics()
        
    def _compute_statistics(self):
        """Compute various statistics for difficulty assessment"""
        # Combine all training triples
        all_triples = np.concatenate(self.train_list)
        
        # Entity frequency
        self.entity_freq = defaultdict(int)
        for triple in all_triples:
            self.entity_freq[triple[0]] += 1
            self.entity_freq[triple[2]] += 1
            
        # Relation frequency
        self.relation_freq = defaultdict(int)
        for triple in all_triples:
            self.relation_freq[triple[1]] += 1
            
        # Entity degree (number of unique relations)
        self.entity_degree = defaultdict(set)
        for triple in all_triples:
            self.entity_degree[triple[0]].add(triple[1])
            self.entity_degree[triple[2]].add(triple[1])
            
    def compute_difficulty_scores(self, metric='frequency'):
        """
        Compute difficulty scores for each time step.
        Lower scores indicate easier samples.
        """
        difficulty_scores = []
        
        for t, snap in enumerate(self.train_list):
            if metric == 'frequency':
                # Based on entity and relation frequency (lower frequency = harder)
                scores = []
                for triple in snap:
                    s, r, o = triple
                    entity_score = min(self.entity_freq[s], self.entity_freq[o])
                    relation_score = self.relation_freq[r]
                    scores.append(1.0 / (1.0 + entity_score + relation_score))
                score = np.mean(scores)
                
            elif metric == 'rarity':
                # Based on entity rarity
                scores = []
                for triple in snap:
                    s, r, o = triple
                    entity_rarity = 1.0 / (1.0 + min(self.entity_freq[s], self.entity_freq[o]))
                    relation_rarity = 1.0 / (1.0 + self.relation_freq[r])
                    scores.append(entity_rarity + relation_rarity)
                score = np.mean(scores)
                
            elif metric == 'temporal_distance':
                # Based on temporal distance from beginning
                score = t / len(self.train_list)
                
            elif metric == 'degree':
                # Based on entity degree complexity
                scores = []
                for triple in snap:
                    s, r, o = triple
                    degree_complexity = len(self.entity_degree[s]) + len(self.entity_degree[o])
                    scores.append(degree_complexity)
                score = np.mean(scores)
                
            else:
                score = 1.0  # Default uniform difficulty
                
            difficulty_scores.append(score)
            
        return np.array(difficulty_scores)

def get_curriculum_samples(train_list, difficulty_scores, current_ratio, strategy='easy_first'):
    """
    Get training samples based on curriculum learning strategy.
    
    Args:
        train_list: List of training snapshots
        difficulty_scores: Difficulty score for each snapshot
        current_ratio: Current ratio of data to include
        strategy: 'easy_first', 'hard_first', or 'mixed'
    """
    n_samples = int(len(train_list) * current_ratio)
    n_samples = max(1, min(n_samples, len(train_list)))
    
    if strategy == 'easy_first':
        # Sort by difficulty (ascending - easiest first)
        sorted_indices = np.argsort(difficulty_scores)
        selected_indices = sorted_indices[:n_samples]
    elif strategy == 'hard_first':
        # Sort by difficulty (descending - hardest first)
        sorted_indices = np.argsort(difficulty_scores)[::-1]
        selected_indices = sorted_indices[:n_samples]
    elif strategy == 'mixed':
        # Mix of easy and hard samples
        sorted_indices = np.argsort(difficulty_scores)
        easy_count = n_samples // 2
        hard_count = n_samples - easy_count
        easy_indices = sorted_indices[:easy_count]
        hard_indices = sorted_indices[-hard_count:]
        selected_indices = np.concatenate([easy_indices, hard_indices])
    else:
        # Random sampling (baseline)
        selected_indices = np.random.choice(len(train_list), n_samples, replace=False)
    
    return sorted(selected_indices)

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
    sr_to_sro = np.load('../data/{}/his_dict_new/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()

    
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
    
    # 文件转储 - file dump
    if mode == "test": # test模式写入，train模式忽略
        filename = '../result/'+ args.dataset + ".csv"
        if os.path.isfile(filename) == False:# If the file does not exist, create it
            with open (filename,'w', newline='') as f:
                # 写入列名 Write column names
                fieldnames=['encoder','opn','pre_type','use_static','use_cl','gpu','datetime','pre_weight',
                            'train_len','test_len','temperature','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        # 写入数据 data input
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

class AdaptiveCurriculumScheduler:
    """
    Adaptive Curriculum Learning scheduler that adjusts curriculum based on model performance.
    """
    
    def __init__(self, strategy='cosine', difficulty_metric='frequency', 
                 start_ratio=0.2, end_ratio=1.0, warmup_epochs=5,
                 adaptation_strategy='loss_based', 
                 patience=3, adaptation_threshold=0.01,
                 acceleration_factor=1.5, deceleration_factor=0.7):
        """
        Args:
            strategy: Base scheduling strategy ('linear', 'exponential', 'step', 'cosine')
            difficulty_metric: Difficulty metric to use
            start_ratio: Initial fraction of training data to use
            end_ratio: Final fraction of training data to use
            warmup_epochs: Number of epochs to reach full curriculum
            adaptation_strategy: 'performance_based', 'loss_based', or 'gradient_based'
            patience: Number of epochs to wait before adapting
            adaptation_threshold: Minimum improvement threshold for adaptation
            acceleration_factor: Factor to speed up curriculum when performance is good
            deceleration_factor: Factor to slow down curriculum when performance is poor
        """
        self.strategy = strategy
        self.difficulty_metric = difficulty_metric
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_epochs = warmup_epochs
        self.adaptation_strategy = adaptation_strategy
        self.patience = patience
        self.adaptation_threshold = adaptation_threshold
        self.acceleration_factor = acceleration_factor
        self.deceleration_factor = deceleration_factor
        
        # Adaptive state tracking
        self.performance_history = []
        self.loss_history = []
        self.gradient_norm_history = []
        self.current_progress_rate = 1.0  # Rate at which curriculum progresses
        self.stagnation_count = 0
        self.last_adaptation_epoch = 0
        
    def update_performance_metrics(self, epoch, mrr_score=None, loss=None, grad_norm=None):
        """Update performance metrics for adaptive scheduling"""
        if mrr_score is not None:
            self.performance_history.append((epoch, mrr_score))
        if loss is not None:
            self.loss_history.append((epoch, loss))
        if grad_norm is not None:
            self.gradient_norm_history.append((epoch, grad_norm))
    
    def _should_adapt(self, epoch):
        """Determine if curriculum should be adapted based on recent performance"""
        if epoch - self.last_adaptation_epoch < self.patience:
            return False, 0.0
            
        if self.adaptation_strategy == 'performance_based':
            return self._performance_based_adaptation(epoch)
        elif self.adaptation_strategy == 'loss_based':
            return self._loss_based_adaptation(epoch)
        elif self.adaptation_strategy == 'gradient_based':
            return self._gradient_based_adaptation(epoch)
        else:
            return False, 0.0
    
    def _performance_based_adaptation(self, epoch):
        """Adapt based on MRR performance improvement"""
        if len(self.performance_history) < self.patience + 1:
            return False, 0.0
            
        recent_scores = [score for _, score in self.performance_history[-self.patience:]]
        older_scores = [score for _, score in self.performance_history[-2*self.patience:-self.patience]]
        
        if len(older_scores) == 0:
            return False, 0.0
            
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        improvement = recent_avg - older_avg
        
        if improvement > self.adaptation_threshold:
            # Good performance - accelerate curriculum
            return True, self.acceleration_factor
        elif improvement < -self.adaptation_threshold:
            # Poor performance - decelerate curriculum
            return True, self.deceleration_factor
        else:
            # Stagnation - maintain current pace
            self.stagnation_count += 1
            return False, 0.0
    
    def _loss_based_adaptation(self, epoch):
        """Adapt based on loss improvement"""
        if len(self.loss_history) < self.patience + 1:
            return False, 0.0
            
        recent_losses = [loss for _, loss in self.loss_history[-self.patience:]]
        older_losses = [loss for _, loss in self.loss_history[-2*self.patience:-self.patience]]
        
        if len(older_losses) == 0:
            return False, 0.0
            
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        improvement = older_avg - recent_avg  # Loss decrease is improvement
        
        if improvement > self.adaptation_threshold:
            return True, self.acceleration_factor
        elif improvement < -self.adaptation_threshold:
            return True, self.deceleration_factor
        else:
            return False, 0.0
    
    def _gradient_based_adaptation(self, epoch):
        """Adapt based on gradient norm stability"""
        if len(self.gradient_norm_history) < self.patience + 1:
            return False, 0.0
            
        recent_norms = [norm for _, norm in self.gradient_norm_history[-self.patience:]]
        gradient_variance = np.var(recent_norms)
        
        # Low variance indicates stable training - can accelerate
        # High variance indicates unstable training - should decelerate
        if gradient_variance < 0.1:  # Stable gradients
            return True, self.acceleration_factor
        elif gradient_variance > 1.0:  # Unstable gradients
            return True, self.deceleration_factor
        else:
            return False, 0.0
    
    def get_current_ratio(self, epoch, total_epochs):
        """Calculate current curriculum ratio with adaptive adjustment"""
        if epoch >= self.warmup_epochs:
            return self.end_ratio
        
        # Check if adaptation is needed
        should_adapt, adaptation_factor = self._should_adapt(epoch)
        
        if should_adapt:
            self.current_progress_rate *= adaptation_factor
            self.last_adaptation_epoch = epoch
            # Clamp progress rate to reasonable bounds
            self.current_progress_rate = max(0.1, min(3.0, self.current_progress_rate))
            print(f"Epoch {epoch}: Adapting curriculum progress rate to {self.current_progress_rate:.3f}")
        
        # Calculate base progress with adaptive rate
        adaptive_progress = min(1.0, (epoch / self.warmup_epochs) * self.current_progress_rate)
        
        # Apply base scheduling strategy
        if self.strategy == 'linear':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * adaptive_progress
        elif self.strategy == 'exponential':
            ratio = self.start_ratio * ((self.end_ratio / self.start_ratio) ** adaptive_progress)
        elif self.strategy == 'step':
            if adaptive_progress < 0.33:
                ratio = self.start_ratio
            elif adaptive_progress < 0.66:
                ratio = (self.start_ratio + self.end_ratio) / 2
            else:
                ratio = self.end_ratio
        elif self.strategy == 'cosine':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * \
                   (1 - np.cos(adaptive_progress * np.pi)) / 2
        else:
            ratio = self.end_ratio
            
        return min(ratio, self.end_ratio)
    
    def get_adaptation_info(self):
        """Return current adaptation state information"""
        return {
            'progress_rate': self.current_progress_rate,
            'stagnation_count': self.stagnation_count,
            'last_adaptation_epoch': self.last_adaptation_epoch,
            'performance_history_length': len(self.performance_history),
            'recent_performance_trend': self._get_recent_trend()
        }
    
    def _get_recent_trend(self):
        """Get recent performance trend"""
        if len(self.performance_history) < 2:
            return "insufficient_data"
        
        recent_scores = [score for _, score in self.performance_history[-min(5, len(self.performance_history)):]]
        if len(recent_scores) >= 2:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            if trend > 0.001:
                return "improving"
            elif trend < -0.001:
                return "declining"
            else:
                return "stable"
        return "stable"


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    global mrr_raw, mrr_filter
    mrr_raw, mrr_filter = [], []
    """
    Modified run_experiment with adaptive curriculum learning.
    """
    # Load configuration for grid search
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # Load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # Initialize ADAPTIVE curriculum learning components
    adaptive_scheduler = AdaptiveCurriculumScheduler(
        strategy=getattr(args, 'curriculum_strategy', 'linear'),
        difficulty_metric=getattr(args, 'difficulty_metric', 'degree'),
        start_ratio=getattr(args, 'curriculum_start_ratio', 0.1),
        end_ratio=getattr(args, 'curriculum_end_ratio', 1.0),
        warmup_epochs=getattr(args, 'curriculum_warmup_epochs', 10),
        adaptation_strategy=getattr(args, 'adaptation_strategy', 'loss_based'),
        patience=getattr(args, 'adaptation_patience', 3),
        adaptation_threshold=getattr(args, 'adaptation_threshold', 0.01),
        acceleration_factor=getattr(args, 'acceleration_factor', 1.5),
        deceleration_factor=getattr(args, 'deceleration_factor', 0.7)
    )
    
    difficulty_analyzer = DifficultyAnalyzer(train_list, num_nodes, num_rels)
    difficulty_scores = difficulty_analyzer.compute_difficulty_scores(
        adaptive_scheduler.difficulty_metric
    )
    
    print(f"Adaptive Curriculum Learning Settings:")
    print(f"  Base Strategy: {adaptive_scheduler.strategy}")
    print(f"  Difficulty Metric: {adaptive_scheduler.difficulty_metric}")
    print(f"  Adaptation Strategy: {adaptive_scheduler.adaptation_strategy}")
    print(f"  Start Ratio: {adaptive_scheduler.start_ratio}")
    print(f"  Adaptation Patience: {adaptive_scheduler.patience}")

    # Load answer lists for evaluation
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    
    model_name = "{}-adaptive-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature,
                args.pre_weight, args.use_cl, args.pre_type, args.n_hidden, args.encoder, str(time.time()))
    model_state_file = '../models/' + 'esh_adaptive'
    
    print("Sanity Check: stat name : {}".format(model_state_file))
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # Static graph setup (same as before)
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # Create model (same as before)
    model = RecurrentRGCN(args.decoder, args.encoder, num_nodes, num_rels, num_static_rels, num_words,
                          args.n_hidden, args.opn, sequence_len=args.train_history_len,
                          num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
                          dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm, input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation, weight=args.weight, pre_weight=args.pre_weight,
                          discount=args.discount, angle=args.angle, use_static=args.add_static_graph,
                          pre_type=args.pre_type, use_cl=args.use_cl, temperature=args.temperature,
                          entity_prediction=args.entity_prediction, relation_prediction=args.relation_prediction,
                          use_cuda=use_cuda, gpu=args.gpu, analysis=args.run_analysis, cl_approach = args.cl_approach)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    print(f"  loss method:",args.cl_approach)

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(model, train_list+valid_list, test_list, num_rels, num_nodes, 
                                   use_cuda, all_ans_list_test, all_ans_list_r_test, model_state_file, 
                                   static_graph, "test")
    else:
        print("-------------- Start training with ADAPTIVE curriculum learning --------------\n")
        best_mrr = 0
        his_best = 0
        avgloss = []
        # Training loop with ADAPTIVE curriculum learning
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []
            epoch_grad_norms = []

            # Get ADAPTIVE curriculum ratio and samples
            current_ratio = adaptive_scheduler.get_current_ratio(epoch, args.n_epochs)
            selected_indices = get_curriculum_samples(
                train_list, 
                difficulty_scores, 
                current_ratio,
                strategy=getattr(args, 'curriculum_sample_strategy', 'mixed')
            )
            
            # Get adaptation info for logging
            adaptation_info = adaptive_scheduler.get_adaptation_info()
            
            print(f"Epoch {epoch}: Using {len(selected_indices)}/{len(train_list)} samples")
            print(f"  Curriculum ratio: {current_ratio:.3f}")
            print(f"  Progress rate: {adaptation_info['progress_rate']:.3f}")
            print(f"  Performance trend: {adaptation_info['recent_performance_trend']}")

            for train_sample_num in tqdm(selected_indices):
                if train_sample_num == 0: 
                    continue
                    
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                # Load subgraph data
                subgraph_arr = np.load('../data/{}/his_graph_for_new/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                subgraph_arr_inv = np.load('../data/{}/his_graph_inv_new/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair = e2r(output[0], num_rels)
                que_pair_inv = e2r(inverse_triples, num_rels)
                
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda() 
                
                for id in range(2): 
                    if id % 2 == 0: 
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair, subg_snap, train_sample_num, 
                                                                             history_glist, triples, static_graph, use_cuda)
                    else:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num, 
                                                                             history_glist, inverse_triples, static_graph, use_cuda)

                    loss = loss_e + loss_static + loss_cl
                
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())
                    
                    loss.backward()
                    
                    # Calculate gradient norm for adaptive scheduling
                    total_grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    epoch_grad_norms.append(total_grad_norm)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            avgloss.append(np.mean(losses))
            # Update adaptive scheduler with performance metrics
            avg_loss = np.mean(losses)
            avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
            adaptive_scheduler.update_performance_metrics(epoch, loss=avg_loss, grad_norm=avg_grad_norm)

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} | Adaptive Ratio: {:.3f} | Best MRR {:.4f}"
                  .format(epoch, avg_loss, np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), current_ratio, best_mrr))

            # Validation with adaptive scheduler update
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, train_list, valid_list, num_rels, num_nodes, use_cuda, 
                                          all_ans_list_valid, all_ans_list_r_valid, model_state_file, static_graph, mode="train")
                
                # Update adaptive scheduler with MRR performance
                adaptive_scheduler.update_performance_metrics(epoch, mrr_score=mrr_filter.item())
                
                if not args.relation_evaluation:
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs or his_best >= 5:
                            break
                    else:
                        his_best = 0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                        
            torch.cuda.empty_cache()
        np.savetxt('lossval.txt', avgloss)
        plt.title('Plot of training loss')
        plt.plot(avgloss)
        plt.savefig('lossfig.png')    
        # Final testing
        mrr_raw, mrr_filter = test(model, train_list+valid_list, test_list, num_rels, num_nodes, use_cuda, 
                                  all_ans_list_test, all_ans_list_r_test, model_state_file, static_graph, mode="test")
    return mrr_raw, mrr_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogCL')

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

    parser.add_argument("--cl_approach", type=str, default="laplace", 
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



