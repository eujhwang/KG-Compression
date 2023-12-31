import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_max, scatter_mean, scatter_add
import torch.nn.functional as F

from sources.kgmoe.gcn_conv import GCNConv
from sources.kgmoe.copooling import CoPooling


class GraphEncoder(nn.Module):
    def __init__(self, embed_size, gamma=0.8, alpha=1, beta=1, aggregate_method="max", tokenizer=None, hop_number=2,
                 num_mixtures=3, assign_ratio=0.5, pool_type="sag", batch_size=None):
        super(GraphEncoder, self).__init__()

        self.hop_number = hop_number

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.aggregate_method = aggregate_method
        self.tokenizer = tokenizer
        self.num_mixtures = num_mixtures
        self.assign_ratio = assign_ratio
        self.relation_embed = nn.Embedding(50, embed_size, padding_idx=0)
        self.pool_type = pool_type
        self.batch_size = batch_size
        print("assign_ratio:", self.assign_ratio, "num_mixtures:", self.num_mixtures, "pool_type:", self.pool_type,
              "batch_size:", self.batch_size)
        # self.triple_linear = nn.Linear(embed_size * 3, embed_size, bias=False)

        self.W_s = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.W_n = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.W_r = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.gate_linear = nn.Linear(embed_size, 1)

        # self.gcn_att = DenseGCNConv(embed_size, 1, bias=True)
        self.lin = Linear(embed_size, 1, bias=False)
        self.bias = Parameter(torch.Tensor(1))

        # self.score_layer = GCNConv(embed_size * (self.hop_number+1), 1)
        self.node_linear = nn.Linear(embed_size * (self.hop_number + 1), embed_size, bias=True)
        if self.pool_type == "sag" or self.pool_type == "sag-h":
            self.score_layer = GCNConv(embed_size, 1, add_self_loops=True)
        if self.pool_type == "copooling":
            self.copooling = CoPooling(embed_size=embed_size, K=10, alpha=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.pool_type == "sag":
            self.score_layer.reset_parameters()
        self.node_linear.reset_parameters()

    def normalize_batch_adj(self, adj):  # adj shape: batch_size * num_node * num_node, D^{-1/2} (A+I) D^{-1/2}
        dim = adj.size()[1]
        A = adj + torch.eye(dim, device=adj.device)
        deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)

        newA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        newA = (adj.sum(-1) > 0).float().unsqueeze(-1).to(adj.device) * newA
        return newA

    def coarsen_graph(self, concept_hidden, relation_hidden, head, tail, relation, triple_label, adj):
        bsz = head.size(0)  # batch_size 4
        # mem_t = head.size(1)  # max_triple_len 600
        mem = concept_hidden.size(1)  # max_concept_length 300
        # hidden_size = concept_hidden.size(2)  # concept hidden size 768

        # concept_hidden (b, n, d) (180, 300, 768) adj (b, n, n)
        x = concept_hidden

        # Parameterization of S part
        out = self.lin(x) # X.W_a
        norm_adj = self.normalize_batch_adj(adj)
        out = torch.matmul(norm_adj, out)
        out = out + self.bias

        # not sure why torch.pow 2
        alpha_vec = F.sigmoid(torch.pow(out, 2)).squeeze()
        # alpha_vec = F.sigmoid(out).squeeze()
        batch_num_nodes = []
        for i in range(bsz):
            batch_num_nodes.append(len(set(head[i, :].tolist())))
        batch_num_nodes = torch.tensor(batch_num_nodes)

        cut_batch_num_nodes = batch_num_nodes
        cut_value = torch.zeros_like(alpha_vec[:, 0])
        assign_ratio = 0.1
        for i in range(bsz):
            if cut_batch_num_nodes[i] > 1:
                cut_batch_num_nodes[i] = torch.ceil(cut_batch_num_nodes[i].float() * assign_ratio) + 1
                temptopk, topk_ind = alpha_vec[i].topk(cut_batch_num_nodes[i], dim=-1)
                cut_value[i] = temptopk[-1]
            else:
                cut_value[i] = 0

        cut_alpha_vec = F.relu(alpha_vec+0.0000001 - torch.unsqueeze(cut_value, -1))
        S = torch.mul(norm_adj, cut_alpha_vec.unsqueeze(1))  # repeat rows of cut_alpha_vec, #b * n * n
        S = F.normalize(S, p=1, dim=-1)

        embedding_tensor = torch.matmul(torch.transpose(S, 1, 2), x)  # equals to torch.einsum('bij,bjk->bik',...)
        new_adj = torch.matmul(torch.matmul(torch.transpose(S, 1, 2), adj), S)  # batched matrix multiply

        # embedding_tensor: [4, 300, 768] new_adj: [4, 300, 300] S: [4, 300, 300]
        return embedding_tensor, new_adj

    def filter_adj(self, edge_index, edge_attr, perm, triple_label, num_nodes=None):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes,), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]
        triple_label = triple_label[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return torch.stack([row, col], dim=0), edge_attr, triple_label

    def sag_pooling(self, concept_hidden, relation_hidden, head, tail, relation, triple_label, concept_labels,
                    concept_ids):
        bsz_mix, num_concepts, hid_dim = concept_hidden.shape
        bsz_mix, num_concepts = concept_ids.shape
        bsz_mix, num_max_triples = head.shape

        if bsz_mix == self.batch_size * self.num_mixtures:
            num_mixtures = self.num_mixtures
        else:
            num_mixtures = 1
        concept_hidden = concept_hidden.view(-1, num_mixtures, num_concepts, hid_dim)
        concept_ids = concept_ids.view(-1, num_mixtures, num_concepts)
        concept_labels = concept_labels.view(-1, num_mixtures, num_concepts)

        head = head.view(-1, num_mixtures, num_max_triples)
        tail = tail.view(-1, num_mixtures, num_max_triples)
        relation = relation.view(-1, num_mixtures, num_max_triples)
        triple_label = triple_label.view(-1, num_mixtures, num_max_triples)
        relation_hidden = relation_hidden.view(-1, num_mixtures, num_max_triples, hid_dim)

        node_repr_list, concept_labels_list, concept_ids_list, perm_list = [], [], [], []
        for mix in range(num_mixtures):
            _concept_hidden = concept_hidden[..., mix, :, :]
            _concept_ids = concept_ids[..., mix, :]
            _concept_labels = concept_labels[..., mix, :]

            _head = head[..., mix, :]
            _tail = tail[..., mix, :]
            _triple_label = triple_label[..., mix, :]

            bsz = _concept_hidden.size(0)  # batch_size 4
            x = _concept_hidden

            new_Xs = []
            new_concept_labels = []
            new_concept_ids = []
            new_perm = []
            for i in range(bsz):
                xi = x[i]
                edge_index = torch.stack([_head[i, :], _tail[i, :]], dim=1).T
                score = self.score_layer(xi, edge_index.to(x.device), triple_label=_triple_label[i]).squeeze()
                label = edge_index.new_zeros(xi.size(0))
                score = score.masked_fill(_concept_labels[i] == -1, float('-inf'))

                perm = topk(score, self.assign_ratio, label)
                score = torch.tanh(score[perm])
                _xi = xi[perm] * score.view(-1, 1)
                # print("perm: {}".format(perm.shape))
                new_perm.append(perm)
                new_concept_label = _concept_labels[i, perm]
                new_concept_id = _concept_ids[i, perm]

                new_Xs.append(_xi)
                new_concept_labels.append(new_concept_label)
                new_concept_ids.append(new_concept_id)
            new_Xs = torch.stack(new_Xs, dim=0).to(x.device)
            new_concept_labels = torch.stack(new_concept_labels, dim=0).to(x.device)
            new_concept_ids = torch.stack(new_concept_ids, dim=0).to(x.device)
            new_perm = torch.stack(new_perm, dim=0).to(x.device)

            node_repr_list.append(new_Xs)
            concept_labels_list.append(new_concept_labels)
            concept_ids_list.append(new_concept_ids)
            perm_list.append(new_perm)

        node_repr = torch.stack(node_repr_list, dim=1).to(x.device)
        concept_labels = torch.stack(concept_labels_list, dim=1).to(x.device)
        concept_ids = torch.stack(concept_ids_list, dim=1).to(x.device)
        perm_idx = torch.stack(perm_list, dim=1).to(x.device)

        # reshape tensors
        node_repr = node_repr.view(-1, node_repr.shape[-2], node_repr.shape[-1])
        concept_labels = concept_labels.view(-1, concept_labels.shape[-1])
        concept_ids = concept_ids.view(-1, concept_ids.shape[-1])

        return node_repr, concept_labels, concept_ids, perm_idx

    def sagh_pooling(self, concept_hidden, relation_hidden, head, tail, relation, triple_label, concept_labels,
                     concept_ids):
        bsz_mix, num_concepts, hid_dim = concept_hidden.shape
        bsz_mix, num_concepts = concept_ids.shape
        bsz_mix, num_max_triples = head.shape

        if bsz_mix == self.batch_size * self.num_mixtures:
            num_mixtures = self.num_mixtures
        else:
            num_mixtures = 1
        concept_hidden = concept_hidden.view(-1, num_mixtures, num_concepts, hid_dim)
        concept_ids = concept_ids.view(-1, num_mixtures, num_concepts)
        concept_labels = concept_labels.view(-1, num_mixtures, num_concepts)

        head = head.view(-1, num_mixtures, num_max_triples)
        tail = tail.view(-1, num_mixtures, num_max_triples)
        relation = relation.view(-1, num_mixtures, num_max_triples)
        triple_label = triple_label.view(-1, num_mixtures, num_max_triples)
        relation_hidden = relation_hidden.view(-1, num_mixtures, num_max_triples, hid_dim)

        node_repr_list, concept_labels_list, concept_ids_list = [], [], []
        head_list, tail_list, rel_list, rel_hid_list, triple_label_list = [], [], [], [], []
        for mix in range(num_mixtures):
            _concept_hidden = concept_hidden[..., mix, :, :]
            _concept_ids = concept_ids[..., mix, :]
            _concept_labels = concept_labels[..., mix, :]

            _head = head[..., mix, :]
            _tail = tail[..., mix, :]
            _triple_label = triple_label[..., mix, :]
            _relation = relation[..., mix, :]
            _relation_hidden = relation_hidden[..., mix, :, :]

            bsz = _concept_hidden.size(0)  # batch_size 4
            x = _concept_hidden

            new_Xs, new_concept_labels, new_concept_ids = [], [], []
            new_heads, new_tails, new_rels, new_rel_hids, new_triple_labels = [], [], [], [], []
            for i in range(bsz):
                xi = x[i]
                edge_index = torch.stack([_head[i, :], _tail[i, :]], dim=1).T
                score = self.score_layer(xi, edge_index.to(x.device), triple_label=_triple_label[i]).squeeze()
                label = edge_index.new_zeros(xi.size(0))
                score = score.masked_fill(_concept_labels[i] == -1, float('-inf'))

                perm = topk(score, self.assign_ratio, label)
                edge_index, new_relation, new_triple_label = self.filter_adj(edge_index, _relation[i, :], perm,
                                                                             _triple_label[i], num_nodes=score.size(0))

                new_head = edge_index[0, :]
                new_tail = edge_index[1, :]
                new_relation_hidden = _relation_hidden[i, new_relation]
                assert new_head.shape == new_tail.shape == new_relation.shape == new_triple_label.shape
                assert new_head.shape[0] == new_relation_hidden.shape[0]

                new_head = F.pad(input=new_head, pad=(0, num_max_triples - new_head.shape[0]), mode='constant', value=0)
                new_tail = F.pad(input=new_tail, pad=(0, num_max_triples - new_tail.shape[0]), mode='constant', value=0)
                new_relation = F.pad(input=new_relation, pad=(0, num_max_triples - new_relation.shape[0]),
                                     mode='constant', value=0)
                new_triple_label = F.pad(input=new_triple_label, pad=(0, num_max_triples - new_triple_label.shape[0]),
                                         mode='constant', value=-1)
                new_relation_hidden = F.pad(input=new_relation_hidden,
                                            pad=(0, 0, 0, num_max_triples - new_relation_hidden.shape[0]),
                                            mode='constant', value=0)

                score = torch.tanh(score[perm])
                _xi = xi[perm] * score.view(-1, 1)

                new_concept_label = _concept_labels[i, perm]
                new_concept_id = _concept_ids[i, perm]

                new_Xs.append(_xi)
                new_concept_labels.append(new_concept_label)
                new_concept_ids.append(new_concept_id)

                new_heads.append(new_head)
                new_tails.append(new_tail)
                new_rels.append(new_relation)
                new_triple_labels.append(new_triple_label)
                new_rel_hids.append(new_relation_hidden)

            new_Xs = torch.stack(new_Xs, dim=0).to(x.device)
            new_concept_labels = torch.stack(new_concept_labels, dim=0).to(x.device)
            new_concept_ids = torch.stack(new_concept_ids, dim=0).to(x.device)
            new_heads = torch.stack(new_heads, dim=0).to(x.device)
            new_tails = torch.stack(new_tails, dim=0).to(x.device)
            new_rels = torch.stack(new_rels, dim=0).to(x.device)
            new_triple_labels = torch.stack(new_triple_labels, dim=0).to(x.device)
            new_rel_hids = torch.stack(new_rel_hids, dim=0).to(x.device)

            node_repr_list.append(new_Xs)
            concept_labels_list.append(new_concept_labels)
            concept_ids_list.append(new_concept_ids)
            head_list.append(new_heads)
            tail_list.append(new_tails)
            rel_list.append(new_rels)
            triple_label_list.append(new_triple_labels)
            rel_hid_list.append(new_rel_hids)

        node_repr = torch.stack(node_repr_list, dim=1).to(x.device)
        concept_labels = torch.stack(concept_labels_list, dim=1).to(x.device)
        concept_ids = torch.stack(concept_ids_list, dim=1).to(x.device)
        head = torch.stack(head_list, dim=1).to(x.device)
        tail = torch.stack(tail_list, dim=1).to(x.device)
        relation = torch.stack(rel_list, dim=1).to(x.device)
        triple_label = torch.stack(triple_label_list, dim=1).to(x.device)
        relation_hidden = torch.stack(rel_hid_list, dim=1).to(x.device)

        # reshape tensors
        node_repr = node_repr.view(-1, node_repr.shape[-2], node_repr.shape[-1])
        concept_labels = concept_labels.view(-1, concept_labels.shape[-1])
        concept_ids = concept_ids.view(-1, concept_ids.shape[-1])

        head = head.view(-1, head.shape[-1])
        tail = tail.view(-1, tail.shape[-1])
        relation = relation.view(-1, relation.shape[-1])
        triple_label = triple_label.view(-1, triple_label.shape[-1])
        relation_hidden = relation_hidden.view(-1, relation_hidden.shape[-2], relation_hidden.shape[-1])

        return node_repr, concept_labels, concept_ids, head, tail, relation, relation_hidden, triple_label

    def multi_layer_comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
        return concept_hidden, relation_hidden

    def comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        relation_hidden: bsz x mem_t x hidden
        '''
        bsz = head.size(0)  # batch_size 4
        mem_t = head.size(1)  # max_triple_len 600
        mem = concept_hidden.size(1)  # max_concept_length 300
        hidden_size = concept_hidden.size(2)  # concept hidden size 768
        # concept_hidden (180, 300, 768)
        update_node = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()  # [4, 300, 768]

        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()  # [4, 600]
        count_out = torch.zeros(bsz, mem).to(head.device).float()  # [4, 300]

        # head [4, 600], tail [4, 600]
        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)

        scatter_add(o, tail, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=1, out=update_node)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=1, out=update_node)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        update_node = act(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)

    def multi_layer_gcn(self, concept_hidden, head, tail, triple_label, layer_number=2):
        for i in range(layer_number):
            concept_hidden = self.gcn(concept_hidden, head, tail, triple_label, i)
        return concept_hidden

    def gcn(self, concept_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        '''
        bsz = head.size(0)
        mem_t = head.size(1)
        mem = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)
        update_hidden = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()
        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()
        count_out = torch.zeros(bsz, mem).to(head.device).float()

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, tail, dim=1, out=update_hidden)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_hidden)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_hidden = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_hidden) / count_out.clamp(min=1).unsqueeze(2)
        update_hidden = act(update_hidden)

        return update_hidden

    def multi_hop(self, triple_prob, distance, head, tail, concept_label, triple_label, gamma=0.8, iteration=3,
                  method="avg"):
        '''
        triple_prob: bsz x L x mem_t
        distance: bsz x mem
        head, tail: bsz x mem_t
        concept_label: bsz x mem
        triple_label: bsz x mem_t

        Init binary vector with source concept == 1 and others 0
        expand to size: bsz x L x mem
        '''
        concept_probs = []
        cpt_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*cpt_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        final_mask = init_mask.clone()

        init_mask.masked_fill_((concept_label == -1).unsqueeze(1), 0)
        concept_probs.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        for _ in range(iteration):
            ''' Calculate triple head score '''
            node_score = concept_probs[-1]
            triple_head_score = node_score.gather(2, head)
            triple_head_score.masked_fill_((triple_label == -1).unsqueeze(1), 0)
            # Method:
            # - avg: s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v)
            # - max: s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
            update_value = triple_head_score * gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)
            out.masked_fill_((concept_label == -1).unsqueeze(1), 0)
            
            concept_probs.append(out)
        
        ''' Natural decay of concept that is multi-hop away from source '''
        total_concept_prob = final_mask * -1e5
        for prob in concept_probs[1:]:
            total_concept_prob += prob
        # bsz x L x mem
        return total_concept_prob

    def forward(self, concept_ids, distance, head, tail, relation, triple_label, mixture_ids=None, concept_labels=None):

        memory = self.embed_word(concept_ids)
        rel_repr = self.relation_embed(relation)

        if mixture_ids is not None:
            mixture_embed = self.mixture_embed(mixture_ids) # [60*3, 300, 768]
            memory = memory + 1.0 * mixture_embed

        if self.pool_type == "sag":
            ###################################### start of sag_pooling ######################################
            concept_hidden = memory
            relation_hidden = rel_repr
            concept_hidden_list = [memory]
            # _concept_hidden = memory
            for i in range(self.hop_number):
                concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
                # _concept_hidden = _concept_hidden + concept_hidden
                concept_hidden_list.append(concept_hidden)
            node_repr = torch.cat(concept_hidden_list, dim=-1).to(memory.device)  # [bsz, #concepts, 768 * num_hop]
            node_repr = self.node_linear(node_repr)
            # node_repr, rel_repr = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, 0)
            pooled_node_repr, concept_labels, concept_ids, perm_idx = self.sag_pooling(node_repr, rel_repr, head, tail, relation,
                                                                      triple_label, concept_labels, concept_ids)
            ###################################### end of sag_pooling ######################################
        # elif self.pool_type == "sag-h":
        #     concept_hidden = memory
        #     relation_hidden = rel_repr
        #     node_repr, rel_repr = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, 0)
        #     node_repr, concept_labels, concept_ids, head, tail, relation, relation_hidden, triple_label = \
        #         self.sagh_pooling(node_repr, rel_repr, head, tail, relation, triple_label, concept_labels, concept_ids)
        #     node_repr, rel_repr = self.comp_gcn(node_repr, relation_hidden, head, tail, triple_label, 1)
        #     pooled_node_repr, concept_labels, concept_ids, head, tail, relation, relation_hidden, triple_label = \
        #         self.sagh_pooling(node_repr, rel_repr, head, tail, relation, triple_label, concept_labels, concept_ids)
        else:
            node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)
            pooled_node_repr = node_repr
            memory = None
            perm_idx = None

        ###################################### start of co_pooling ######################################
        # concept_hidden = memory
        # relation_hidden = rel_repr
        # concept_hidden_list = [memory]
        # for i in range(self.hop_number):
        #     concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
        #     concept_hidden_list.append(concept_hidden)
        # node_repr = torch.cat(concept_hidden_list, dim=-1).to(memory.device)  # [bsz, #concepts, 768 * num_hop]
        # node_repr = self.node_linear(node_repr)
        #
        # if self.pool_type == "copooling":
        #     node_repr, concept_labels, concept_ids = self.copooling(concept_hidden, relation_hidden, head, tail,
        #                                                             triple_label, concept_labels, concept_ids)
        ###################################### end of co_pooling ######################################

        ###################################### start of Coarsening ######################################
        # coarse_x = memory
        # for i in range(3):
        #     coarse_x, adj = self.coarsen_graph(coarse_x, rel_repr, head, tail, relation, triple_label, adj)
        #
        # bsz = head.shape[0] // self.num_mixtures
        # head = torch.full([bsz, head.shape[1]], -1).to(memory.device)
        # tail = torch.full([bsz, tail.shape[1]], -1).to(memory.device)
        # skipped = 0
        # for i in range(bsz):
        #     tmp_head = adj[i].nonzero()[:, 0]
        #     tmp_tail = adj[i].nonzero()[:, 1]
        #
        #     if len(tmp_head) > head.shape[1]:
        #         skipped += 1
        #         continue
        #     if len(tmp_tail) > tail.shape[1]:
        #         continue
        #     # print("tmp_head:", tmp_head.shape, tmp_head[:20])
        #     # print("tmp_tail:", tmp_tail.shape, tmp_tail[:20])
        #     head[i, :len(tmp_head)] = tmp_head
        #     tail[i, :len(tmp_tail)] = tmp_tail
        # if skipped > 0:
        #     print("skipped:", skipped)
        # expand_size = bsz, self.num_mixtures, head.shape[-1]
        # head = head.unsqueeze(1).expand(*expand_size).contiguous().view(bsz * self.num_mixtures, head.shape[-1])
        # tail = tail.unsqueeze(1).expand(*expand_size).contiguous().view(bsz * self.num_mixtures, tail.shape[-1])
        # node_repr = self.multi_layer_gcn2(coarse_x, rel_repr, head, tail, triple_label, layer_number=self.hop_number)
        ###################################### end of Coarsening ######################################

        ###################################### original ######################################
        # node_repr = concept outputs
        # node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)

        # head_repr = torch.gather(node_repr, 1, head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        # tail_repr = torch.gather(node_repr, 1, tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        #
        # # bsz x mem_triple x hidden
        # triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        ###################################### end of original ######################################

        # concept_ids is needed for generating step.
        return pooled_node_repr.to(concept_ids.device), memory, concept_labels, concept_ids, perm_idx

    def generate(self, src_input_ids, attention_mask, src_position_ids, 
                    concept_ids, concept_label, distance, 
                    head, tail, relation, triple_label,
                    vocab_map, map_mask,
                    seq_generator):

        memory = self.word_embed(concept_ids)

        rel_repr = self.relation_embd(relation)

        node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)

        head_repr = torch.gather(node_repr, 1, head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        
        # bsz x mem_triple x hidden
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        
        sample = {"input_ids": src_input_ids, "attention_mask": attention_mask, "position_ids": src_position_ids}
        memory = {"triple_repr": triple_repr,
                  "distance": distance,
                  "head": head,
                  "tail": tail,
                  "concept_label": concept_label,
                  "triple_label": triple_label,
                  "vocab_map": vocab_map,
                  "map_mask": map_mask}

        return seq_generator.generate(self.autoreg_forward, sample, memory)

    def autoreg_forward(self, input_ids, attention_mask, position_ids, memory_dict, do_generate=False, lm_mask=None):

        hidden_states = self.transformer(input_ids, attention_mask=attention_mask,
                                         position_ids=position_ids)[0]

        if do_generate:
            hidden_states = hidden_states[:, -1, :].unsqueeze(1)

        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)
        triple_logits = torch.matmul(hidden_states, self.triple_linear(memory_dict["triple_repr"]).transpose(1, 2))
        
        triple_score = sigmoid(triple_logits)
        # bsz x L x mem_t
    
        triple_score = triple_score.masked_fill((memory_dict["triple_label"] == -1).unsqueeze(1), 0)

        # aggregate probability to nodes
        unorm_cpt_probs = self.multi_hop(triple_score,
                                         memory_dict["distance"],
                                         memory_dict["head"],
                                         memory_dict["tail"],
                                         memory_dict["concept_label"],
                                         memory_dict["triple_label"],
                                         gamma=self.gamma,
                                         iteration=self.hop_number,
                                         method=self.aggregate_method)
        # bsz x L x mem
        cpt_probs = softmax(unorm_cpt_probs)
        # bsz x L x mem

        cpt_probs_vocab = cpt_probs.gather(2, memory_dict["vocab_map"].unsqueeze(1).expand(cpt_probs.size(0), cpt_probs.size(1), -1))

        cpt_probs_vocab.masked_fill_((memory_dict["map_mask"] == 0).unsqueeze(1), 0)
        # bsz x L x vocab
        
        gate = sigmoid(self.gate_linear(hidden_states))
        # bsz x L x 1
        
        lm_logits = self.lm_head(hidden_states)
        lm_probs = softmax(lm_logits)
        
        if do_generate:
            hybrid_probs = lm_probs * (1 - gate) + gate * cpt_probs_vocab
        else:
            hybrid_probs = lm_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(1) * cpt_probs_vocab

        return hybrid_probs, gate, triple_score