import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import transpose, coalesce


def filter_edge_index(edge_index_i, triple_label_i, edge_weight_i=None):
    if (triple_label_i == -1).any().item():
        perm = (triple_label_i == -1).nonzero().shape[0]
        edge_index_i = edge_index_i[:, :-perm] # there are some cases the sentence has no concepts in KG. edge_index_i.nelement() == 0

        if edge_weight_i is not None:
            edge_weight_i = edge_weight_i[:-perm]

    return edge_index_i, edge_weight_i

class PageRank(MessagePassing):
    def __init__(self, K, alpha, **kwargs):
        super(PageRank, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        # PPR-like
        temp = alpha * (1 - alpha) ** np.arange(K + 1)
        temp[-1] = (1 - alpha) ** K
        self.temp = Parameter(torch.tensor(temp))

    def forward(self, concept_hidden, relation_hidden, head, tail, triple_label):
        bsz = concept_hidden.shape[0]
        edge_index = torch.stack([head, tail], dim=1)  # [48, 2, 600]
        edge_weight = torch.ones([bsz, edge_index.shape[-1]]).to(concept_hidden.device) # TODO: replace edge_weight to real weights!!

        hiddens = []
        for i in range(bsz):
            xi = concept_hidden[i]
            edge_index_i = edge_index[i]
            edge_weight_i = edge_weight[i]
            triple_label_i = triple_label[i]

            edge_index_i, edge_weight_i = filter_edge_index(edge_index_i, triple_label_i, edge_weight_i)
            edge_index_i, norm_i = gcn_norm(edge_index=edge_index_i, edge_weight=edge_weight_i, num_nodes=xi.size(0),
                                            add_self_loops=True, dtype=concept_hidden.dtype)

            xi = concept_hidden[i]
            hidden_i = xi * (self.temp[0])
            for k in range(self.K):
                xi = self.propagate(edge_index_i, x=xi, norm=norm_i)
                gamma = self.temp[k + 1]
                hidden_i = hidden_i + gamma * xi
            hiddens.append(hidden_i)
        hidden = torch.stack(hiddens, dim=0)  # [48, 300, 768]
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GraphAttention(torch.nn.Module):
    # reference: https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L324
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, dropout_prob=0.6, log_attention_weights=False):
        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self, concept_hidden, head, tail):
        #
        # Step 1: Linear Projection + regularization
        #

        # concept_hiddn = [# concepts, hid_dim]
        in_nodes_features = concept_hidden  # unpack data
        # num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        # bsz = in_nodes_features.shape[0]

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features) # [48, 300, 1, 768]

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1) # [48, 300, 1]
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1) # [180, 300, 1, 768] * [180, 1, 1, 768] -> [48, 300, 1]

        # edge_index = torch.stack([head, tail], dim=1)
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted = self.lift(scores_source, scores_target, head, tail) # [48, 600], [48, 600]
        scores_per_edge = scores_source_lifted + scores_target_lifted

        return torch.sigmoid(scores_per_edge)

    def lift(self, scores_source, scores_target, head, tail):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """

        bsz = head.shape[0]
        src_nodes_index = head
        trg_nodes_index = tail

        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)

        return scores_source, scores_target

class CoPooling(nn.Module):
    # reference for GAT code: https://github.com/PetarV-/GAT
    # reference for generalized pagerank code: https://github.com/jianhao2016/GPRGNN
    def __init__(self, embed_size, K, alpha):
        super(CoPooling, self).__init__()
        self.pagerank = PageRank(K, alpha)
        self.graph_attn = GraphAttention(num_in_features=embed_size,
                                         num_out_features=embed_size, num_of_heads=1)
        self.linear = nn.Linear(2 * embed_size, embed_size)

    def forward(self, concept_hidden, relation_hidden, head, tail, triple_label, concept_labels, concept_ids):
        x_cut = self.pagerank(concept_hidden, relation_hidden, head, tail, triple_label)

        edge_index = torch.stack([head, tail], dim=1)
        bsz = head.shape[0]
        num_nodes = concept_hidden.shape[1]

        self.edge_ratio = 0.8
        self.node_ratio = 0.7

        concept_hiddens = []
        new_concept_labels = []
        new_concept_ids = []
        for i in range(bsz):
            triple_label_i = triple_label[i]
            edge_index_i = edge_index[i]

            edge_index_i, _ = filter_edge_index(edge_index_i, triple_label_i, edge_weight_i=None)
            if edge_index_i.nelement() == 0:
                concept_hiddens.append(concept_hidden[i][:int(num_nodes*0.7)])
                new_concept_labels.append(concept_labels[i][:int(num_nodes*0.7)])
                new_concept_ids.append(concept_ids[i][:int(num_nodes*0.7)])
                continue

            head_i = edge_index_i[0]
            tail_i = edge_index_i[1]
            attn_i = self.graph_attn(x_cut[i], head_i, tail_i).squeeze(-1)  # [1, 600] edge attention

            assert head_i.shape == tail_i.shape == attn_i.shape

            edge_index_t_i, attn_t_i = transpose((head_i, tail_i), attn_i, num_nodes, num_nodes)
            edge_tmp = torch.cat((edge_index_i, edge_index_t_i), 1)
            att_tmp = torch.cat((attn_i, attn_t_i), 0)
            edge_index_i, attn_i = coalesce(edge_tmp, att_tmp, num_nodes, num_nodes, 'mean') # [2, 600], [1200]

            attn_i_np = attn_i.cpu().data.numpy()
            cut_val = np.percentile(attn_i_np, int(100 * (1 - self.edge_ratio)))  # this is for keep the top edge_ratio edges

            attn_i = attn_i * (attn_i >= cut_val)  # keep the edge_ratio higher weights of edges
            kep_idx = attn_i > 0.0
            cut_edge_index_i, cut_edge_attr_i = edge_index_i[:, kep_idx], attn_i[kep_idx]  # [2, 840], [840]


            # row, col = cut_edge_index_i
            # deg = scatter_add(cut_edge_attr_i, row, dim=0, dim_size=num_nodes)
            # deg_inv_sqrt = deg.pow(-0.5)
            # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            #
            # expand_deg = torch.zeros((cut_edge_attr_i.size(0),), device=edge_index.device)
            # expand_deg[-num_nodes:] = torch.ones((num_nodes,), device=edge_index.device)
            #
            # norm_attn_i = expand_deg - deg_inv_sqrt[row] * cut_edge_attr_i * deg_inv_sqrt[col] # [1200]
            # norm_attn_i = attn_i

            # propagate
            x_j = concept_hidden[i][cut_edge_index_i[0]]
            x_j = cut_edge_attr_i.view(-1, 1) * x_j
            index = cut_edge_index_i[1]
            out = scatter_add(x_j, index, dim=0, dim_size=num_nodes)

            score = torch.sum(torch.abs(out), dim=1) + 0.0000001
            label = cut_edge_index_i.new_zeros(concept_hidden[i].size(0))

            perm = topk(score, self.node_ratio, label)
            x_topk = concept_hidden[i][perm]

            attention_dense = (
                to_dense_adj(cut_edge_index_i, edge_attr=cut_edge_attr_i, max_num_nodes=num_nodes)).squeeze()
            concept_hidden_i = torch.matmul(attention_dense[perm], concept_hidden[i])
            concept_hidden_i = torch.cat([x_topk, concept_hidden_i], dim=-1)
            concept_hidden_i = self.linear(concept_hidden_i)

            concept_label = concept_labels[i, perm]
            concept_id = concept_ids[i, perm]

            concept_hiddens.append(concept_hidden_i)
            new_concept_labels.append(concept_label)
            new_concept_ids.append(concept_id)
        concept_hidden = torch.stack(concept_hiddens, dim=0) # [180, 210, 768]
        concept_labels = torch.stack(new_concept_labels, dim=0).to(concept_hidden.device) # [180, 210]
        concept_ids = torch.stack(new_concept_ids, dim=0).to(concept_hidden.device) # [180, 210]

        return concept_hidden, concept_labels, concept_ids