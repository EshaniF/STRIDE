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
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for Temporal Knowledge Graph Completion
    """
    def __init__(self, curriculum_type="difficulty", start_ratio=0.3, end_ratio=1.0, 
                 curriculum_epochs=10, total_epochs=100):
        """
        Args:
            curriculum_type: Type of curriculum ('difficulty', 'temporal', 'frequency', 'combined')
            start_ratio: Starting ratio of training samples
            end_ratio: Ending ratio of training samples (usually 1.0)
            curriculum_epochs: Number of epochs to gradually increase samples
            total_epochs: Total training epochs
        """
        self.curriculum_type = curriculum_type
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.curriculum_epochs = curriculum_epochs
        self.total_epochs = total_epochs
        
    def get_sample_ratio(self, epoch):
        """Get the ratio of samples to use at current epoch"""
        if epoch >= self.curriculum_epochs:
            return self.end_ratio
        
        # Linear increase from start_ratio to end_ratio
        progress = epoch / self.curriculum_epochs
        return self.start_ratio + (self.end_ratio - self.start_ratio) * progress
    
    def get_curriculum_indices(self, train_list, epoch, difficulty_scores=None):
        """
        Get indices of training samples based on curriculum strategy
        """
        sample_ratio = self.get_sample_ratio(epoch)
        total_samples = len(train_list) - 1  # Exclude first sample (index 0)
        num_samples = int(total_samples * sample_ratio)
        
        if self.curriculum_type == "temporal":
            # Start with earlier time steps, gradually add later ones
            indices = list(range(1, min(num_samples + 1, len(train_list))))
            
        elif self.curriculum_type == "difficulty" and difficulty_scores is not None:
            # Start with easier samples (lower difficulty scores)
            sorted_indices = sorted(range(1, len(train_list)), 
                                  key=lambda i: difficulty_scores[i])
            indices = sorted_indices[:num_samples]
            
        elif self.curriculum_type == "frequency":
            # Start with high-frequency entities/relations
            indices = self._get_frequency_based_indices(train_list, num_samples)
            
        elif self.curriculum_type == "combined":
            # Combine temporal and difficulty
            temporal_weight = 0.7
            difficulty_weight = 0.3
            indices = self._get_combined_indices(train_list, num_samples, 
                                               difficulty_scores, temporal_weight, difficulty_weight)
        else:
            # Random sampling as baseline
            indices = list(range(1, min(num_samples + 1, len(train_list))))
            
        return sorted(indices)
    
    def _get_frequency_based_indices(self, train_list, num_samples):
        """Get indices based on entity/relation frequency"""
        entity_freq = defaultdict(int)
        relation_freq = defaultdict(int)
        
        # Count frequencies
        for i, snap in enumerate(train_list[1:], 1):
            for triple in snap:
                entity_freq[triple[0]] += 1
                entity_freq[triple[2]] += 1
                relation_freq[triple[1]] += 1
        
        # Calculate sample scores (higher frequency = easier)
        sample_scores = []
        for i, snap in enumerate(train_list[1:], 1):
            score = 0
            for triple in snap:
                score += entity_freq[triple[0]] + entity_freq[triple[2]] + relation_freq[triple[1]]
            sample_scores.append((i, score / len(snap)))
        
        # Sort by score (descending) and take top samples
        sample_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sample_scores[:num_samples]]
    
    def _get_combined_indices(self, train_list, num_samples, difficulty_scores, 
                            temporal_weight, difficulty_weight):
        """Combine temporal and difficulty-based curriculum"""
        if difficulty_scores is None:
            return list(range(1, min(num_samples + 1, len(train_list))))
        
        # Normalize scores
        temporal_scores = np.array(range(1, len(train_list)))
        temporal_scores = (temporal_scores - temporal_scores.min()) / (temporal_scores.max() - temporal_scores.min())
        
        diff_scores = np.array([difficulty_scores[i] for i in range(1, len(train_list))])
        diff_scores = (diff_scores - diff_scores.min()) / (diff_scores.max() - diff_scores.min())
        
        # Combined score (lower is easier)
        combined_scores = temporal_weight * temporal_scores + difficulty_weight * diff_scores
        
        # Sort and select
        sorted_indices = sorted(range(1, len(train_list)), 
                              key=lambda i: combined_scores[i-1])
        return sorted_indices[:num_samples]


class DifficultyEstimator:
    """
    Estimate difficulty of training samples for curriculum learning
    """
    def __init__(self):
        self.entity_degrees = defaultdict(int)
        self.relation_frequencies = defaultdict(int)
        self.temporal_patterns = defaultdict(list)
    
    def compute_difficulty_scores(self, train_list, num_nodes, num_rels):
        """
        Compute difficulty scores for each training sample
        Lower scores indicate easier samples
        """
        self._analyze_graph_statistics(train_list)
        difficulty_scores = {}
        
        for i, snap in enumerate(train_list):
            if i == 0:
                continue
                
            # Metrics for difficulty
            avg_entity_degree = np.mean([self.entity_degrees[triple[0]] + self.entity_degrees[triple[2]] 
                                       for triple in snap])
            
            rare_relation_ratio = sum(1 for triple in snap 
                                    if self.relation_frequencies[triple[1]] < 10) / len(snap)
            
            temporal_volatility = self._compute_temporal_volatility(snap, i)
            
            # Combine metrics (normalize and weight)
            difficulty = (
                0.3 * (1.0 / (1.0 + avg_entity_degree / 100)) +  # Higher degree = easier
                0.4 * rare_relation_ratio +  # More rare relations = harder
                0.3 * temporal_volatility  # More volatility = harder
            )
            
            difficulty_scores[i] = difficulty
            
        return difficulty_scores
    
    def _analyze_graph_statistics(self, train_list):
        """Analyze graph statistics for difficulty estimation"""
        for snap in train_list:
            for triple in snap:
                self.entity_degrees[triple[0]] += 1
                self.entity_degrees[triple[2]] += 1
                self.relation_frequencies[triple[1]] += 1
                self.temporal_patterns[triple[1]].append(len(train_list))
    
    def _compute_temporal_volatility(self, snap, time_idx):
        """Compute temporal volatility as a difficulty measure"""
        volatilities = []
        for triple in snap:
            rel_pattern = self.temporal_patterns[triple[1]]
            if len(rel_pattern) > 1:
                # Compute variance in temporal occurrences
                volatility = np.var(rel_pattern) / (time_idx + 1)
                volatilities.append(volatility)
        
        return np.mean(volatilities) if volatilities else 0.0


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
    

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    global mrr_raw, mrr_filter
    mrr_raw, mrr_filter = [], []
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

    # Initialize curriculum learning components
    curriculum_scheduler = CurriculumScheduler(
        curriculum_type=getattr(args, 'curriculum_type', 'temporal'),
        start_ratio=getattr(args, 'curriculum_start_ratio', 0.3),
        end_ratio=1.0,
        curriculum_epochs=getattr(args, 'curriculum_epochs', args.n_epochs // 3),
        total_epochs=args.n_epochs)
    difficulty_estimator = DifficultyEstimator()
    difficulty_scores = None
    
    if curriculum_scheduler.curriculum_type in ['difficulty', 'combined']:
        print("Computing difficulty scores for curriculum learning...")
        difficulty_scores = difficulty_estimator.compute_difficulty_scores(
            train_list, num_nodes, num_rels
        )

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature,args.pre_weight, args.use_cl, args.pre_type,  args.n_hidden, args.encoder,str(time.time()))
    model_state_file = '../models/' + 'esh' 
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


    # create stat
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
                        pre_weight = args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type = args.pre_type,
                        use_cl = args.use_cl,
                        temperature = args.temperature,
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

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter= test(model,
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
        best_mrr = 0
        #his_best tracks how many evaluations have passed without improvement for early stopping.
        his_best = 0
        avgloss= []
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

             # Get curriculum-based training indices
            curriculum_indices = curriculum_scheduler.get_curriculum_indices(
                train_list, epoch, difficulty_scores
            )
            
            print(f"Epoch {epoch}: Using {len(curriculum_indices)}/{len(train_list)-1} training samples "
                  f"(ratio: {len(curriculum_indices)/(len(train_list)-1):.2f})")
            # idx = [_ for _ in range(len(train_list))]

            # for train_sample_num in tqdm(idx):
            for train_sample_num in tqdm(curriculum_indices):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]

                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # subgraph_arr = np.load('../data/{}/his_graph_for/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                # subgraph_arr_inv = np.load('../data/{}/his_graph_inv/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                # subgraph_arr = np.load('../data/{}/his_graph_for_new_0_3/train_s_r_{}.npy'.format(args.dataset, train_sample_num), allow_pickle=True)
                # subgraph_arr_inv = np.load('../data/{}/his_graph_inv_new_0_3/train_o_r_{}.npy'.format(args.dataset, train_sample_num), allow_pickle=True)
                subgraph_arr = np.load('../data/{}/his_graph_for_new/train_s_r_{}.npy'.format(args.dataset, train_sample_num), allow_pickle=True)
                subgraph_arr_inv = np.load('../data/{}/his_graph_inv_new/train_o_r_{}.npy'.format(args.dataset, train_sample_num), allow_pickle=True)
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)   #Take out the sampling subgraph
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair =  e2r(output[0], num_rels)
                que_pair_inv =  e2r(inverse_triples, num_rels)
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda() 
                for id in range(2): 
                    if id %2 ==0: 
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair, subg_snap, train_sample_num, history_glist, triples, static_graph, use_cuda)
                    else:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num, history_glist, inverse_triples,static_graph, use_cuda)

                    loss = loss_e+ loss_static +loss_cl
                
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                # break
            avgloss.append(np.mean(losses))
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            # validation
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
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            print('the break statement 1 stopped the training')
                            break
                        if his_best>=5:
                            print('the break statement 2 stopped the training')
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            torch.cuda.empty_cache()
        np.savetxt('lossval.txt', avgloss)
        plt.title('Plot of training loss')
        plt.plot(avgloss)
        plt.savefig('lossfig.png')

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
    parser.add_argument("--pre-type",  type=str, default="all",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=True,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    """Add curriculum learning arguments to argument parser"""
    parser.add_argument('--use_curriculum', action='store_true', 
                       help='Use curriculum learning')
    parser.add_argument('--curriculum_type', type=str, default='frequency',
                       choices=['temporal', 'difficulty', 'frequency', 'combined'],
                       help='Type of curriculum learning strategy')
    parser.add_argument('--curriculum_start_ratio', type=float, default=0.2,
                       help='Starting ratio of training samples in curriculum')
    parser.add_argument('--curriculum_epochs', type=int, default=15,
                       help='Number of epochs for curriculum phase (default: n-epochs//3)')                  
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



