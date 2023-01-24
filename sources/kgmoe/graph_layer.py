import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_scatter import scatter_max, scatter_mean, scatter_add
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from trainers.kgtrainer_utils import sinkhorn_loss_default


from torch_geometric.nn import DenseGCNConv, GCNConv
# from torch_geometric.nn.dense.linear import Linear

class GraphEncoder(nn.Module):
    def __init__(self, embed_size, gamma=0.8, alpha=1, beta=1, aggregate_method="max", tokenizer=None, hop_number=2, num_mixtures=3):
        super(GraphEncoder, self).__init__()

        self.hop_number = hop_number
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.aggregate_method = aggregate_method
        self.tokenizer = tokenizer
        self.num_mixtures = num_mixtures

        self.relation_embed = nn.Embedding(50, embed_size, padding_idx=0)
        
        # self.triple_linear = nn.Linear(embed_size * 3, embed_size, bias=False)
        
        self.W_s = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)]) 
        self.W_n = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.W_r = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.gate_linear = nn.Linear(embed_size, 1)

        # self.gcn_att = DenseGCNConv(embed_size, 1, bias=True)
        self.lin = Linear(embed_size, 1, bias=False)
        self.bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

        self.score_layer = GCNConv(embed_size * (self.hop_number+1), 1)
        self.node_linear = nn.Linear(embed_size * (self.hop_number+1), embed_size, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def normalize_batch_adj(self, adj):  # adj shape: batch_size * num_node * num_node, D^{-1/2} (A+I) D^{-1/2}
        dim = adj.size()[1]
        A = adj + torch.eye(dim, device=adj.device)
        deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)

        newA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        newA = (adj.sum(-1)>0).float().unsqueeze(-1).to(adj.device) * newA
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

    def sag_pooling(self, concept_hidden, relation_hidden, head, tail, relation, triple_label, adj):
        bsz = head.size(0)  # batch_size 4
        # mem_t = head.size(1)  # max_triple_len 600
        mem = concept_hidden.size(1)  # max_concept_length 300
        # hidden_size = concept_hidden.size(2)  # concept hidden size 768

        # concept_hidden (b, n, d) (180, 300, 768) adj (b, n, n)
        x = concept_hidden

        assign_ratio = 0.7
        new_Xs = []
        for i in range(bsz):
            xi = x[i]
            edge_index = torch.stack([head[i, :], tail[i, :]], dim=1).T
            score = self.score_layer(xi, edge_index.to(x.device)).squeeze()
            label = edge_index.new_zeros(xi.size(0))

            perm = topk(score, assign_ratio, label)
            _xi = xi[perm] * F.relu(score[perm]).view(-1, 1)
            new_xi = torch.zeros(xi.shape, device=x.device)
            new_xi[perm] = _xi
            # edge_index, edge_attr = filter_adj(edge_index, relation_hidden[i], perm, num_nodes=score.size(0))

            new_xi = self.node_linear(new_xi)
            new_Xs.append(new_xi)
        node_repr = torch.stack(new_Xs, dim=0).to(x.device)
        return node_repr

    def multi_layer_comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_number=2):
        concept_hidden_list = []
        for i in range(layer_number):
            concept_hidden_list.append(concept_hidden)
            concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
        concept_hidden_list.append(concept_hidden)
        concept_hidden = torch.cat(concept_hidden_list, dim=-1)
        return concept_hidden, relation_hidden

    def comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        relation_hidden: bsz x mem_t x hidden
        '''
        bsz = head.size(0)      # batch_size 4
        mem_t = head.size(1)    # max_triple_len 600
        mem = concept_hidden.size(1) # max_concept_length 300
        hidden_size = concept_hidden.size(2) # concept hidden size 768
        # concept_hidden (180, 300, 768)
        update_node = torch.zeros_like(concept_hidden).to(concept_hidden.device).float() # [4, 300, 768]

        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float() # [4, 600]
        count_out = torch.zeros(bsz, mem).to(head.device).float() # [4, 300]

        # head [4, 600], tail [4, 600]
        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)

        scatter_add(o, tail, dim=1, out=update_node)
        scatter_add( - relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=1, out=update_node)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_node)
        scatter_add( - relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=1, out=update_node)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        update_node = act(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)

    def multi_layer_gcn2(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_number=2):
        for i in range(layer_number):
            concept_hidden = self.gcn2(concept_hidden, relation_hidden, head, tail, triple_label, i)
        return concept_hidden

    def gcn2(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        relation_hidden: bsz x mem_t x hidden
        '''
        # print("adj:", adj.shape, adj)
        bsz = head.size(0)      # batch_size 4
        mem_t = head.size(1)    # max_triple_len 600
        mem = concept_hidden.size(1) # max_concept_length 300
        hidden_size = concept_hidden.size(2) # concept hidden size 768
        # concept_hidden (180, 300, 768)
        update_node = torch.zeros_like(concept_hidden).to(concept_hidden.device).float() # [4, 300, 768]
        # count_out = torch.zeros(bsz, mem).to(adj.device).float()

        # for i in range(bsz):
        #     head = adj[i].nonzero()[:, 0]
        #     tail = adj[i].nonzero()[:, 1]
        #
        #     count = torch.ones_like(head).to(head.device).float()
        #
        #     o = concept_hidden[i].gather(0, head.unsqueeze(-1).expand(head.shape[0], hidden_size)) # [b, # of head noes, d] [60, 89, 768]
        #     scatter_add(o, tail, dim=0, out=update_node[i])
        #     scatter_add(count, tail, dim=0, out=count_out[i])
        #
        #     o = concept_hidden[i].gather(0, tail.unsqueeze(-1).expand(tail.shape[0], hidden_size))
        #     scatter_add(o, head, dim=0, out=update_node[i])
        #     scatter_add(count, head, dim=0, out=count_out[i])
        # act = nn.ReLU()
        # update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        # update_node = act(update_node)

        # print("update_node:", update_node.shape, update_node)
        count = torch.ones_like(head).to(head.device).masked_fill_(head == -1, 0).float() # [4, 600]
        count_out = torch.zeros((concept_hidden.shape[0], concept_hidden.shape[1])).to(head.device).float() # [4, 300]

        # head [4, 600], tail [4, 600]
        _head = head.unsqueeze(2)
        _tail = tail.unsqueeze(2)
        _head = _head.masked_fill(head.unsqueeze(2) == -1, 0)
        _tail = _tail.masked_fill(tail.unsqueeze(2) == -1, 0)

        o = concept_hidden.gather(1, _head.expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(head.unsqueeze(2) == -1, 0)
        scatter_add(o, _tail.squeeze(2), dim=1, out=update_node)
        scatter_add(count, _tail.squeeze(2), dim=1, out=count_out)

        o = concept_hidden.gather(1, _tail.expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(tail.unsqueeze(2) == -1, 0)
        scatter_add(o, _head.squeeze(2), dim=1, out=update_node)
        scatter_add(count, _head.squeeze(2), dim=1, out=count_out)
        # print("concept_hidden:", concept_hidden.shape, "head:", head.shape, "update_node:", update_node.shape, "count_out:", count_out.shape)
        act = nn.ReLU()
        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        update_node = act(update_node)

        # return update_node, self.W_r[layer_idx](relation_hidden)
        return update_node

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

    def multi_hop(self, triple_prob, distance, head, tail, concept_label, triple_label, gamma=0.8, iteration = 3, method="avg"):
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

    def forward(self, concept_ids, distance, head, tail, relation, triple_label, mixture_ids=None, adj=None):
        
        memory = self.embed_word(concept_ids)
        rel_repr = self.relation_embed(relation)

        if mixture_ids is not None:
            mixture_embed = self.mixture_embed(mixture_ids) # [60*3, 300, 768]
            memory = memory + 1.0 * mixture_embed

        ###################################### start of sag_pooling ######################################
        # concept_hidden = memory
        # relation_hidden = rel_repr
        # concept_hidden_list = [memory]
        # # relation_hidden_list = [rel_repr]
        # for i in range(self.hop_number):
        #     concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
        #     concept_hidden_list.append(concept_hidden)
        #     # relation_hidden_list.append(relation_hidden)
        node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)
        print("node_repr:", node_repr.shape)
        # node_repr = torch.cat(concept_hidden_list, dim=-1).to(memory.device) # [bsz, #concepts, 768 * num_hop]
        node_repr = self.sag_pooling(node_repr, rel_repr, head, tail, relation, triple_label, adj)
        ###################################### end of sag_pooling ######################################

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

        # node_repr = memory + node_repr
        # head_repr = torch.gather(node_repr, 1, head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        # tail_repr = torch.gather(node_repr, 1, tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        #
        # # bsz x mem_triple x hidden
        # triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        ###################################### end of original ######################################

        return node_repr.to(memory.device), memory

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

        hidden_states = self.transformer(input_ids, attention_mask = attention_mask, 
                                                    position_ids = position_ids)[0]

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
                                                gamma = self.gamma,
                                                iteration = self.hop_number,
                                                method = self.aggregate_method)
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