import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

class LightAggregator(nn.Module):
    def __init__(self):
        super(LightAggregator, self).__init__()

    def forward(self, user_emb, entity_emb, interact_mat):
        entity_agg = torch.sparse.mm(interact_mat.t(), user_emb)
        user_agg = torch.sparse.mm(interact_mat, entity_emb)
        return entity_agg, user_agg

class LightGCN(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(LightGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        for i in range(n_hops):
            self.convs.append(LightAggregator())

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, interact_mat, 
                mess_dropout=True, node_dropout=False):
                
        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb
        user_res_emb = user_emb

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](user_emb, entity_emb, interact_mat)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb, dim=-1)
            user_emb = F.normalize(user_emb, dim=-1)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, entity_emb, relation_emb, scores, edge_index, edge_type):

        # [n, d]
        n_entities = entity_emb.shape[0]
        n_relations = relation_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        # scores = torch.sum(entity_emb[head] * relation_emb[edge_type - 1] * entity_emb[tail], dim=-1, keepdim=True)
        # scores = scatter_softmax(src=scores, index=head, dim=0)
        neigh_emb = scores.unsqueeze(-1) * (relation_emb[edge_type - 1] * entity_emb[tail])
        entity_agg = scatter_sum(src=neigh_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        for i in range(n_hops):
            self.convs.append(Aggregator())

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]
    
    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, entity_emb, relation_emb, edge_index, edge_type, 
                mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        entity_res_emb = entity_emb

        head, tail = edge_index
        kg_score = torch.sum(entity_emb[head] * relation_emb[edge_type - 1] * entity_emb[tail], dim=-1)
        kg_score = scatter_softmax(src=kg_score, index=head, dim=0)

        for i in range(len(self.convs)):
            entity_emb = self.convs[i](entity_emb, relation_emb, kg_score, edge_index, edge_type)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb, dim=-1)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)

        return entity_res_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, interact_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.alpha1 = args_config.alpha1
        self.alpha2 = args_config.alpha2
        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.n_factors = args_config.n_factors

        initializer = torch.nn.init.xavier_uniform_
        user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        entity_embed = initializer(torch.empty(self.n_entities, self.emb_size))
        relation_embed = initializer(torch.empty(self.n_relations - 1, self.emb_size))
        latent_embed = initializer(torch.empty(self.n_factors, self.emb_size))
        latent_weight = initializer(torch.empty(self.n_factors, self.emb_size, self.emb_size))

        if args_config.pretrain == 1:
            pretrain_data = torch.load(args_config.data_path + args_config.model_path)
            user_embed.data = pretrain_data['user_para']
            entity_embed[:self.n_items].data = pretrain_data['item_para']
            print('with pretraining.')
        else:
            print('without pretraining.')
        
        self.user_embed = nn.Parameter(user_embed)
        self.entity_embed = nn.Parameter(entity_embed)
        self.relation_embed = nn.Parameter(relation_embed)
        self.latent_embed = nn.Parameter(latent_embed)
        self.latent_weight = nn.Parameter(latent_weight)

        self.edge_index, self.edge_type = self._get_edges(graph)
        # interact_mat = self._convert_sp_mat_to_sp_tensor(interact_mat).to(self.device)
        self.interact_mats = []
        for i in range(self.n_factors):
            interact_sp_tensor = self._convert_sp_mat_to_sp_tensor(interact_mat).to(self.device)
            self.interact_mats.append(interact_sp_tensor)
        self.interaction = self._get_interaction(interact_mat).to(self.device)

        self.gcn_list, self.light_gcn_list = [], []
        for i in range(self.n_factors):
            self.gcn_list.append(GraphConv(n_hops=self.context_hops,
                                           node_dropout_rate=self.node_dropout_rate,
                                           mess_dropout_rate=self.mess_dropout_rate))
            
            self.light_gcn_list.append(LightGCN(n_hops=self.context_hops,
                                                node_dropout_rate=self.node_dropout_rate,
                                                mess_dropout_rate=self.mess_dropout_rate))
        
        self.gate_list = []
        for i in range(self.n_factors):
            self.gate_list.append(nn.Sequential(
                                    nn.Linear(self.emb_size, self.emb_size, bias=True),
                                    nn.Sigmoid()
                                    ).to(self.device))
        
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True)
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True)
                )

    def _get_interaction(self, X):
        i = torch.LongTensor([X.row, X.col])
        return i

    def _convert_sp_mat_to_sp_tensor(self, X):
        # coo = X.tocoo()
        i = torch.LongTensor([X.row, X.col])
        v = torch.from_numpy(X.data).float()
        return torch.sparse_coo_tensor(i, v, X.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)
    
    def _neg_sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.matmul(z1, z2.transpose(1, 2)).sum(-1)
    
    def _pos_sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.sum(z1 * z2, dim=-1)
    
    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _create_graph_contrastive_loss(self, emb_1, emb_2, idx):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        
        emb_c1 = emb_1[idx]
        emb_c2 = emb_2[idx]
        pos = f(self._pos_sim(emb_c1, emb_c2)).diag()
        neg_1 = f(self._neg_sim(emb_c1, emb_c2))
        neg_2 = f(self._neg_sim(emb_c2, emb_c1))
        loss_1 = - torch.log(pos / neg_1)
        loss_2 = - torch.log(pos / neg_2)

        ret = loss_1 + loss_2
        ret = ret.mean()
        return ret
    
    def _create_graph_contrastive_loss2(self, emb_1):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        
        emb_1 = self.fc1(emb_1)
        pos = f(self._pos_sim(emb_1, emb_1)).diag()
        neg_1 = f(self._neg_sim(emb_1, emb_1))
        loss_1 = - torch.log(pos / neg_1)

        ret = loss_1
        ret = ret.mean()
        return ret
    
    def _create_view_contrastive_loss(self, emb_list1, emb_list2, idx):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)

        ret = 0.
        for i in range(self.n_factors):
            emb_1 = self.fc2(emb_list1[i][idx])
            emb_2 = self.fc2(emb_list2[i][idx])
            sim = f(self._sim(emb_1, emb_2))
            loss_1 = - torch.log(sim.diag() / sim.sum(1))
            loss_2 = - torch.log(sim.diag() / sim.sum(0))
            ret += loss_1.mean() + loss_2.mean()

        ret = ret / self.n_factors
        return ret
    
    def _create_bpr_loss(self, users, items, scores, labels, contra_loss_g, contra_loss_v):
        batch_size = scores.shape[0]
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # contra
        contra_loss = self.alpha1 * contra_loss_g + self.alpha2 * contra_loss_v
        
        return bce_loss + emb_loss + contra_loss, scores, bce_loss, emb_loss, \
                contra_loss_g, contra_loss_v
    
    def forward(self, batch=None):
        user = batch['users']
        item = batch['items']
        labels = batch['labels']
        batch_size = user.shape[0]

        self.entity_cf_emb_list, self.user_cf_emb_list, self.entity_kg_emb_list = [], [], []
        user_emb_list, entity_emb_list, rel_emb_list = [], [], []

        for i in range(self.n_factors):
            mapped_user_emb = self.user_embed * self.gate_list[i](self.user_embed)
            mapped_entity_emb = self.entity_embed * self.gate_list[i](self.entity_embed)
            mapped_relation_emb = self.relation_embed * self.gate_list[i](self.relation_embed)
            rel_emb_list.append(mapped_relation_emb)
            user_emb_list.append(mapped_user_emb)
            entity_emb_list.append(mapped_entity_emb)
        
        self.mapped_relations_embs = torch.stack(rel_emb_list, dim=1)
        self.mapped_user_embs = torch.stack(user_emb_list, dim=1)
        self.mapped_entity_embs = torch.stack(entity_emb_list, dim=1)

        for i in range(self.n_factors):
            entity_cf_emb, user_cf_emb = self.light_gcn_list[i](self.mapped_user_embs[:,i,:],
                                                                self.mapped_entity_embs[:,i,:],
                                                                self.interact_mats[i],
                                                                mess_dropout=self.mess_dropout,
                                                                node_dropout=self.node_dropout)

            entity_kg_emb = self.gcn_list[i](self.mapped_entity_embs[:,i,:],
                                            self.mapped_relations_embs[:,i,:],
                                            self.edge_index,
                                            self.edge_type,
                                            mess_dropout=self.mess_dropout,
                                            node_dropout=self.node_dropout)
            
            self.entity_cf_emb_list.append(entity_cf_emb)
            self.user_cf_emb_list.append(user_cf_emb)
            self.entity_kg_emb_list.append(entity_kg_emb)
            
        entity_cf_embs = torch.stack(self.entity_cf_emb_list, dim=1)
        user_cf_embs = torch.stack(self.user_cf_emb_list, dim=1)
        entity_kg_embs = torch.stack(self.entity_kg_emb_list, dim=1)
        
        # [batch_size, k, d]
        i_e_cf = entity_cf_embs[item]
        u_e_cf = user_cf_embs[user]
        i_e_kg = entity_kg_embs[item]

        # predict
        u_e = torch.concat((u_e_cf, u_e_cf), dim=-1)
        i_e = torch.concat((i_e_cf, i_e_kg), dim=-1)
        scores = torch.mean(torch.sigmoid(torch.sum(u_e * i_e, dim=-1)), dim=-1)

        # disentangled contrastive loss
        contra_loss_g = self._create_graph_contrastive_loss2(self.mapped_relations_embs)
        contra_loss_g += self._create_graph_contrastive_loss2(self.mapped_user_embs)
        contra_loss_g += self._create_graph_contrastive_loss2(self.mapped_entity_embs)
      
        # cross-view contrastive loss
        contra_loss_v = self._create_view_contrastive_loss(self.entity_cf_emb_list, self.entity_kg_emb_list, item)

        return self._create_bpr_loss(u_e, i_e, scores, labels, contra_loss_g, contra_loss_v)

    def generate(self):
        self.entity_cf_emb_list, self.user_cf_emb_list, self.entity_kg_emb_list = [], [], []
        user_emb_list, entity_emb_list, rel_emb_list = [], [], []

        for i in range(self.n_factors):
            mapped_user_emb = self.user_embed * self.gate_list[i](self.user_embed)
            mapped_entity_emb = self.entity_embed * self.gate_list[i](self.entity_embed)
            mapped_relation_emb = self.relation_embed * self.gate_list[i](self.relation_embed)
            rel_emb_list.append(mapped_relation_emb)
            user_emb_list.append(mapped_user_emb)
            entity_emb_list.append(mapped_entity_emb)
        
        self.mapped_relations_embs = torch.stack(rel_emb_list, dim=1)
        self.mapped_user_embs = torch.stack(user_emb_list, dim=1)
        self.mapped_entity_embs = torch.stack(entity_emb_list, dim=1)

        for i in range(self.n_factors):
            entity_cf_emb, user_cf_emb = self.light_gcn_list[i](self.mapped_user_embs[:,i,:],
                                                                self.mapped_entity_embs[:,i,:],
                                                                self.interact_mats[i],
                                                                mess_dropout=False,
                                                                node_dropout=False)

            entity_kg_emb = self.gcn_list[i](self.mapped_entity_embs[:,i,:],
                                            self.mapped_relations_embs[:,i,:],
                                            self.edge_index,
                                            self.edge_type,
                                            mess_dropout=False,
                                            node_dropout=False)
            
            self.entity_cf_emb_list.append(entity_cf_emb)
            self.user_cf_emb_list.append(user_cf_emb)
            self.entity_kg_emb_list.append(entity_kg_emb)

        entity_cf_embs = torch.stack(self.entity_cf_emb_list, dim=1)
        user_cf_embs = torch.stack(self.user_cf_emb_list, dim=1)
        entity_kg_embs = torch.stack(self.entity_kg_emb_list, dim=1)

        entity_embs = torch.concat((entity_cf_embs, entity_kg_embs), dim=-1)
        user_embs = torch.concat((user_cf_embs, user_cf_embs), dim=-1)
        return entity_embs, user_embs
    
    def generate_scores(self, batch, entity_embs, user_embs):
        user = batch['users']
        item = batch['items']
        batch_size = user.shape[0]

        u_e = user_embs[user]
        i_e = entity_embs[item]
        scores = torch.mean(torch.sigmoid(torch.sum(u_e * i_e, dim=-1)), dim=-1)
        return scores
    
    def update_interact_mats(self):
        user, item = self.interaction[0], self.interaction[1]
        score_list = []
        for i in range(self.n_factors):
            scores = torch.sum(self.user_cf_emb_list[i][user] * self.entity_kg_emb_list[i][item], dim=-1)
            score_list.append(scores)
        scores = torch.stack(score_list, dim=1)
        scores = nn.Softmax(dim=1)(scores)
        for i in range(self.n_factors):
            self.interact_mats[i] = torch.sparse_coo_tensor(self.interaction, scores[:,i], (self.n_users, self.n_entities)).to(self.device)
            

