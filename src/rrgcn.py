import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, RGAT, UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from collections import defaultdict

class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')




class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, 
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1,pre_weight=0.7, discount=0, angle=0, use_static=False, pre_type = 'short', 
                 use_cl= False, temperature=0.007, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, num_temporal_scales=3):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim #Hidden dimension size for embeddings
        self.layer_norm = layer_norm #Whether to use layer normalization (default: False)
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation #Method to aggregate embeddings ('cat' for concatenation)
        self.relation_evolve = False
        self.weight = weight
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type #Prediction type (default: 'short')
        self.use_cl = use_cl #Whether to use contrastive learning (default: False)
        self.temp =temperature #Temperature parameter for contrastive loss
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.num_temporal_scales = num_temporal_scales

        #linear transformation layers w1, w2, .. w7
        self.w1 = nn.Linear(self.h_dim*2, self.h_dim) #w1: Projects concatenated embeddings (dimension 2h) to hidden dimension h
        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w6 = nn.Linear(self.h_dim,self.h_dim)
        self.w7 = nn.Linear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim*2, self.h_dim)

        # Hierarchical attention components
        self.temporal_scale_attention = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(num_temporal_scales)])
        
        # Cross-scale attention mechanism
        self.scale_attention = nn.Linear(h_dim, num_temporal_scales)
        
        # Scale-specific transformation layers
        self.scale_transforms = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(num_temporal_scales)])
        
        # Temporal scale integration layer
        self.scale_integration = nn.Linear(h_dim * num_temporal_scales, h_dim)

        #Creates trainable parameters for temporal encoding. 
        #These parameters are used to generate time-dependent embeddings through cosine encoding.
        ##edit
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.use_fourier_encoding = True
        self.time_encoding_scales = nn.Parameter(torch.randn(h_dim // 2), requires_grad=True)
        self.time_encoding_shifts = nn.Parameter(torch.randn(h_dim // 2), requires_grad=True)
        ##end edit

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        self.bias_r = nn.Parameter(torch.zeros(1))

        #Creates trainable relation embeddings for all relations (including inverse relations, hence num_rels * 2). 
        #Initializes them using Xavier normal initialization for better training.
        #Note the num_rels * 2 accounts for both forward and inverse relations.
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        #Creates trainable entity embeddings for all entities. These are initialized with normal distribution.
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static: #if using static graph
            #Creates word embeddings (initialized with Xavier normal)
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            #Creates an RGCN layer for processing the static graph and Sets up MSE loss for static graph learning
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        #loss functions for relation prediction and entity prediction tasks
        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        #main Relational GCN cell for processing the dynamic graph. This processes entity embeddings considering different relation types.
        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        #another RGCN cell variant, specifically for historical information processing.
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        #Relational Graph Attention Network
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)

        #simple multi-layer perceptron for projection
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)

        #time-aware gating mechanism, Weight matrix initialized with Xavier uniform
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))

        #Bias vector initialized with zeros
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)  

        #prediction gate weight parameters
        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))
        # self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim))
        # nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))                      

        # GRU cell for relation evolving
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            #entity prediction
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.decoder_ob1 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            #relation prediction
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    #edit
    def fourier_time_encoding(self, t, h_dim):
        """
        Generates Fourier feature encoding for timestamps
        
        Args:
            t (torch.Tensor): Time values
            h_dim (int): Dimensionality of the encoding
        
        Returns:
            torch.Tensor: Time-based Fourier feature encoding
        """
        # Ensure t is a tensor and moved to the correct device
        t = torch.as_tensor(t, dtype=torch.float32).to(self.gpu)
        
        # Split dimensions for sine and cosine components
        dim_half = h_dim // 2
        
        # Create frequency scales (learned parameters)
        scales = self.time_encoding_scales.unsqueeze(0)  # [1, dim_half]
        shifts = self.time_encoding_shifts.unsqueeze(0)  # [1, dim_half]
        
        # Compute Fourier features
        # Learned scaling allows adaptive frequency selection
        scaled_time = t.unsqueeze(-1) * scales
        
        # Add learned shifts to introduce more flexibility
        shifted_time = scaled_time + shifts
        
        # Compute sine and cosine components
        sin_components = torch.sin(shifted_time)
        cos_components = torch.cos(shifted_time)
        
        # Concatenate sine and cosine to create full encoding
        time_encoding = torch.cat([sin_components, cos_components], dim=-1)
        
        return time_encoding

    ##end edit
    def forward(self,sub_graph,T_idx, query_mask, g_list, static_graph, use_cuda):
        #T_idx: Likely a time index, query_mask: A mask for query entities
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            #Concatenates dynamic embeddings and word embeddings and assigns them as node features ('h') in the static graph. 
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束: The evolved representation satisfies the static graph constraints of wordemb
            #RGCN layer to the static graph
            self.statci_rgcn_layer(static_graph, [])
            #Extracts the updated node embeddings from the static graph, keeping only the entity embeddings (first num_ents rows)
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            #Normalizes the static embeddings if layer normalization is enabled.
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            #Sets the current hidden state to the static embeddings
            self.h = static_emb
        else:
            #If not using static graph, uses normalized or raw dynamic embeddings as hidden state and sets static_emb to None
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        #-----------------全局历史建模 - Global history modeling -------------------------------------
        #Applies a GCN to model historical entity information, returning entity representations and subgraph indices.
        self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)
        #Normalizes relation embeddings for historical modeling.
        his_r_emb = F.normalize(self.emb_rel)
        #Computes attention weights by applying a linear transformation to the sum of query mask and historical entity embeddings, followed by softmax
        his_att = F.softmax(self.w5(query_mask+ self.his_ent),dim=1)
        #Applies attention weights to historical entity embeddings.
        his_emb = his_att*self.his_ent
        #Normalizes the attention-weighted historical embeddings.
        his_emb = F.normalize(his_emb)

        history_embs = []
        att_embs = []
        his_temp_embs =[]
        his_rel_embs =[]
        if self.pre_type=="all":
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)
                #Calculates a time value based on the graph's position in the list.
                t2 = len(g_list)-i+1
                #Creates time embeddings using cosine encodings for each entity.
                #edit
                # h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
                h_t = self.fourier_time_encoding(t2, self.h_dim).repeat(self.num_ents, 1)
                #end edit
                #Concatenates current hidden state with time embeddings and applies a linear transformation.
                self.h =self.w4(torch.concat([self.h,h_t],dim=1))
                #Gathers embeddings for edges in the current graph.
                g.r_to_e = g.r_to_e.type(torch.LongTensor)
                temp_e = self.h[g.r_to_e]
                #Initializes a tensor for relation embeddings
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                #Iterates through relation spans and their indices.
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    #Extracts embeddings for the current relation span.
                    x = temp_e[span[0]:span[1],:]
                    #Computes the mean embedding for the current relation.
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    #Assigns the mean embedding to the corresponding relation index.
                    x_input[r_idx] = x_mean
                #Adds the original relation embeddings to the computed relation representations.
                x_input = self.emb_rel + x_input
                #Applies RGCN to the current graph using current hidden state and relation embeddings.
                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])
                #Normalizes the result if layer normalization is enabled.
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                # current_h1 = F.sigmoid(self.w6(current_h))   # 让相应的维度大小早）0~1之间，通过mask矩阵获取query time 出现的实体，其他实体设置为0
                #Set the corresponding dimension size to between 0 and 1, obtain the entities that appear at query time through the mask matrix, and set other entities to 0
                #Computes attention weights over current embeddings.
                att_e = F.softmax(self.w2(query_mask+current_h),dim=1)
                
                #Special handling for the first graph in the sequence.
                if i == 0:
                    #Updates hidden state using an entity cell (GRU)
                    self.h_0 = self.entity_cell(current_h, self.h)    # 第1层输入: Layer 1 Input
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                    # self.hr = self.relation_cell(x_input, self.emb_rel)    # 第1层输入 - layer 1 input
                    # self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                else: #For subsequent graphs in the sequence.
                    #Updates hidden state with the entity cell.
                    self.h_0 = self.entity_cell(current_h, self.h_0)  # 第2层输出==下一时刻第一层输入 : The output of the second layer == the input of the first layer at the next timestamp
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                    # self.hr = self.relation_cell(x_input, self.hr)  # 第2层输出==下一时刻第一层输入:  : The output of the second layer == the input of the first layer at the next timestamp
                    # self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                #Computes time-dependent gate weights using a sigmoid on a linear transformation.
                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                #Updates relation embeddings using a gating mechanism between current and original embeddings.
                self.hr = time_weight * x_input + (1-time_weight) * self.emb_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                #Updates the current hidden state.
                self.h = self.h_0
                #Applies attention weights to the current hidden state.
                att_emb = att_e*self.h_0 
                #Stores the attention-weighted embeddings.
                att_embs.append(att_emb.unsqueeze(0))
            #Computes the mean of all attention-weighted embeddings.
            att_ent = torch.mean(torch.concat(att_embs,dim=0),dim=0)
            att_ent = F.normalize(att_ent)
            #Combines attention embeddings with the most recent historical embeddings.
            history_emb=  att_ent+history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            #If prediction type is not "all", sets relation and history embeddings to None.
            self.hr = None
            history_emb = None

        return history_emb, static_emb, self.hr, his_emb, his_r_emb,his_temp_embs,his_rel_embs


    def predict(self,que_pair, sub_graph,T_id, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            #que_pair: A tuple containing query information
            all_triples = test_triplets
            
            #-----------------Query data processing-------------------------------------
            uniq_e = que_pair[0] #uniq_e: Unique entity IDs
            r_len = que_pair[1] #r_len: Length information for relations
            r_idx = que_pair[2] #r_idx: Relation indices
            #Retrieves relation embeddings for the specified relation indices.
            temp_r = self.emb_rel[r_idx]
            #Creates a tensor to store entity inputs, with zeros for all entities.
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            #This effectively pools all relation embeddings connected to each entity.
            for span, e_idx in zip(r_len, uniq_e): #For each entity and its relation span
                #Extracts relation embeddings for the current span
                x = temp_r[span[0]:span[1],:]
                #Computes the mean embedding across the span
                x_mean = torch.mean(x, dim=0, keepdim=True)
                #Assigns this mean embedding to the corresponding entity in e_input
                e_input[e_idx] = x_mean

            #Creates a query mask tensor, filled with zeros.
            query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            #Retrieves dynamic entity embeddings for the unique entities.
            e1_emb = self.dynamic_emb[uniq_e]
            #Gets the relation embeddings for unique entities
            rel_emb = e_input[uniq_e] #实体所连的所有关系池化: Pooling of all relationships connected to an entity
            #Concatenates entity and relation embeddings and applies a linear transformation to create query embeddings.
            query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
            #Updates the query mask by placing query embeddings at positions corresponding to unique entities.
            query_mask[uniq_e] = query_emb

            #embedding: Entity embeddings, r*emb: Relation embeddings, 
            #his_emb: Historical entity embeddings, his_r_emb: Historical relation embeddings
            embedding, _, r_emb, his_emb, his_r_emb,_,_ = self.forward(sub_graph,T_id, query_mask,test_graph, static_graph, use_cuda)

            if self.pre_type == "all":
                #Uses the object decoder to compute scores for all triplets using the generated embeddings. 
                scores_ob,_= self.decoder_ob.forward( embedding,r_emb, all_triples,  his_emb, self.pre_weight, self.pre_type)
                #Applies softmax to convert raw scores to probabilities across dimension 1.
                score_seq = F.softmax(scores_ob, dim=1)
                #Assigns the softmax scores to score_en (entity scores).
                score_en =score_seq
            #Takes the natural logarithm of entity scores, converting probabilities to log probabilities.
            scores_en = torch.log(score_en)
            return all_triples, scores_en


    def get_loss(self,que_pair, sub_graph,T_idx, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        #Initializes entity predict loss, contrastive learning loss, relation predict loss and static graph loss components as zero tensors
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        
        all_triples = triples

        ### --------------查询数据处理 - Query data processing-----------------------
        uniq_e = que_pair[0] #Unique entity IDs
        r_len = que_pair[1] #Length information for relations
        r_idx = que_pair[2] #Relation indices
        #Retrieves relation embeddings for the specified relation indices.
        temp_r = self.emb_rel[r_idx]
        #Creates a tensor to store entity inputs, with zeros for all entities
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        #For each entity and its relation span
        for span, e_idx in zip(r_len, uniq_e):
            #Extracts relation embeddings for the current span
            x = temp_r[span[0]:span[1],:]
            #Computes the mean embedding across the span
            x_mean = torch.mean(x, dim=0, keepdim=True)
            #Assigns this mean embedding to the corresponding entity in e_input
            e_input[e_idx] = x_mean
        
        #Creates a query mask tensor, filled with zeros
        query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        #Converts the time index to a tensor
        t1 = torch.tensor(T_idx).cuda().to(self.gpu)
        #Creates time embeddings using cosine encoding, setting time value to 0 (representing current time). 
        #The encoding is repeated for all entities.
        #edit
        # q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents,1)
        q_t = self.fourier_time_encoding(t1, self.h_dim).repeat(self.num_ents,1)
        #end edit
        #Concatenates dynamic entity embeddings with time embeddings and applies a linear transformation.
        qe_emb = self.w4(torch.concat([self.dynamic_emb,q_t],dim=1))
        #Retrieves time-aware entity embeddings for the unique entities.
        e1_emb = qe_emb[uniq_e]
        #Retrieves relation embeddings for unique entities.
        rel_emb = e_input[uniq_e] 
        #Concatenates time-aware entity embeddings and relation embeddings, 
        #then applies a linear transformation to create query embeddings.
        query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1)) 
        #Updates the query mask by placing query embeddings at positions corresponding to unique entities.
        query_mask[uniq_e] = query_emb
        #his_temp_embs: Temporal entity embeddings at different timesteps, his_rel_embs: Temporal relation embeddings at different timesteps

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, use_cuda)



        if self.pre_type == "all":
            #Uses the object decoder to compute scores for triples using the generated embeddings.
            scores_ob, _= self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb,self.pre_weight, self.pre_type)
            #Applies softmax to convert raw scores to probabilities.
            score_seq = F.softmax(scores_ob, dim=1)
            #Assigns softmax scores to entity scores.
            score_en = score_seq

        #Takes the logarithm of entity scores, converting probabilities to log probabilities.
        scores_en = torch.log(score_en)
        #Calculates negative log likelihood loss between predicted scores and true object entities (third column of triples).
        loss_ent += F.nll_loss(scores_en, triples[:, 2])

        #Checks if relation prediction is enabled
        if self.relation_prediction:
            #Uses the relation decoder to compute relation scores and reshapes the output to match the number of relation types 
            #(including inverse relations).
            score_rel = self.rdecoder.forward(embedding,r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            #Calculates cross-entropy loss between predicted relation scores and true relation indices (second column of triples).
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        
        if self.use_cl and self.pre_type=="all":
            #Iterates through historical temporal embeddings.
            for id, evolve_emb in enumerate(his_temp_embs):
                #Calculates a time offset based on the position in the historical embeddings.
                t3 = len(his_temp_embs)-id+1
                #Creates query representations by concatenating historical entity embeddings for subject entities and historical relation embeddings for relation types
                query = torch.concat([self.his_ent[all_triples[:, 0]],his_r_emb[all_triples[:, 1]]],dim=1)
                #Creates alternative query representations using temporal embeddings at the current timestep.
                query2 = torch.concat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]],dim=1)
                #Applies the same linear transformation to both query representations.
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                #Calculates contrastive loss between the two transformed query representations 
                #and adds it to the total contrastive loss.
                loss_cl += self.get_loss_conv(x1, x2) 

        return loss_ent, loss_rel, loss_static, loss_cl
    #edit
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to maintain compatibility
        """
        # Get the original state dictionary
        state = super().state_dict(destination, prefix, keep_vars)
        
        # If Fourier encoding is not used, remove these keys to prevent errors
        if not self.use_fourier_encoding:
            state.pop(prefix + 'time_encoding_scales', None)
            state.pop(prefix + 'time_encoding_shifts', None)
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle different state dict configurations
        """
        # Remove Fourier encoding keys if not present or if not using Fourier encoding
        if not self.use_fourier_encoding:
            state_dict.pop('time_encoding_scales', None)
            state_dict.pop('time_encoding_shifts', None)
        
        # Fallback to original parameters if Fourier encoding keys are missing
        if strict and (
            'time_encoding_scales' not in state_dict or 
            'time_encoding_shifts' not in state_dict
        ):
            strict = False
        
        # Load state dictionary with adjusted strictness
        return super().load_state_dict(state_dict, strict)
    ##end edit
    def all_GCN(self,ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        #This assigns the entity embeddings (ent_emb) to the node features in the subgraph
        sub_graph.ndata['h'] = ent_emb 
        #
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        #This creates an index tensor subg_index that Generates a sequence from 0 to (number of nodes - 1), 
        #Creates a boolean mask selecting only nodes with in-degree > 0 (nodes that have incoming edges), 
        #Returns a 1D tensor containing only the indices of nodes with incoming connections
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb),subg_index
    
    def get_loss_conv(self, ent1_emb, ent2_emb):

        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        #This is a standard technique in contrastive learning frameworks to map embeddings to a space where the contrastive loss is applied
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        #Similarity between each element in z1 with all elements in z2
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        #Self-similarity matrix for elements in z1
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)
        # train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)) / 2
        train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)+loss_fn(pred3 / self.temp, labels) + loss_fn(pred4 / self.temp, labels)) / 4
        return train_cl_loss