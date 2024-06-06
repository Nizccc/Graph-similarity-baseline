import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool


import math
import torch_geometric.nn as gnn
from SAT_layers.layers import TransformerEncoderLayer, AttentionModule, NTNNetwork, SEAttentionModule
from einops import repeat
from SAT_layers.layers import SETensorNetworkModule as TensorNetworkModule
import torch.nn.functional as F
from utils import get_config
from torch_geometric.nn.glob import global_add_pool, global_mean_pool

class MLPModule_first(torch.nn.Module):
    def __init__(self, args):
        super(MLPModule_first, self).__init__()
        self.args = args

        self.lin0 = torch.nn.Linear(self.args.num_features, self.args.nhid) #!
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self.args.nhid, self.args.nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self.args.nhid , self.args.nhid)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))  
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = F.relu(self.lin2(scores))

        return scores

class MLPModule_last(torch.nn.Module):
    def __init__(self, args):
        super(MLPModule_last, self).__init__()
        self.args = args

        self.lin0 = torch.nn.Linear(self.args.nhid * 2, self.args.nhid) #!
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self.args.nhid, self.args.nhid // 2)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self.args.nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))  
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = torch.sigmoid(self.lin2(scores)).view(-1)

        return scores
    

class gnnbasedModel(nn.Module):
    def __init__(self, args):
        super(gnnbasedModel, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.num_features = args.num_features   
        self.filters = args.gnn_filters    #!
        self.gnn_list = nn.ModuleList()
        self.num_filter = len(self.filters)

        gnn_enc=args.gnn_encoder   #!
        if gnn_enc == 'GCN':  # append
            self.gnn_list.append(GCNConv(self.num_features, self.filters[0]))
            for i in range(self.num_filter - 1):  # num_filter = 3    i = 0,1
                self.gnn_list.append(GCNConv(self.filters[i], self.filters[i + 1]))
        elif gnn_enc == 'GAT':
            self.gnn_list.append(GATConv(self.num_features, self.filters[0]))
            for i in range(self.num_filter - 1):  # num_filter = 3    i = 0,1
                self.gnn_list.append(GATConv(self.filters[i], self.filters[i + 1]))
        elif gnn_enc == 'GIN':
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.num_features, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ), eps=True))

            for i in range(self.num_filter - 1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.filters[i], self.filters[i + 1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filters[i + 1], self.filters[i + 1]),
                    torch.nn.BatchNorm1d(self.filters[i + 1]),
                ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        
        self.mlp_last = MLPModule_last(self.args)


    def convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p=self.args.dropout, training=self.training)  #!
        return feat     
    
    def global_aggregation_info(self, batch, feat, size=None):
        size = (batch[-1].item() + 1 if size is None else size)  # pairsnum of one batch
        if self.args.pooling == 'add':   #!
           pool = global_add_pool(feat, batch, size=size)  
        elif self.args.pooling == 'mean':   #!
           pool = global_mean_pool(feat,batch,size=size) 
        return pool

    def forward(self, data):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        

        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
      
        # Graph Convolution Operation
        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)
        for i in range(self.num_filter):  # gnn  D-64 64-64 64-64
            conv_source_1 = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)  
            conv_source_2 = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)
        
        global_agg_1 = self.global_aggregation_info( batch_1, conv_source_1)  #sum pooling
        global_agg_2 = self.global_aggregation_info( batch_2, conv_source_2) 

        scores = torch.cat([global_agg_1, global_agg_2], dim=1)  # concatenation
        scores = self.mlp_last(scores)  # mlp  2*64-64 64-32 32-1

        return scores


class mlpbasedModel(nn.Module):
    def __init__(self, args):
        super(mlpbasedModel, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.num_features = args.num_features   
        self.filters = args.gnn_filters    #!
        self.gnn_list = nn.ModuleList()
        self.num_filter = len(self.filters)

        self.mlp_first = MLPModule_first(self.args)
        self.mlp_last = MLPModule_last(self.args)
  
    
    def global_aggregation_info(self, batch, feat, size=None):
        size = (batch[-1].item() + 1 if size is None else size)  # pairsnum of one batch
        if self.args.pooling == 'add':   #!
           pool = global_add_pool(feat, batch, size=size)  
        elif self.args.pooling == 'mean':   #!
           pool = global_mean_pool(feat,batch,size=size) 
        return pool

    def forward(self, data):
        
        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
        mlp_source_1 = self.mlp_first(features_1)   #mlp:  D-64 64-64 64-64 
        mlp_source_2 = self.mlp_first(features_2)
        
        global_agg_1 = self.global_aggregation_info( batch_1, mlp_source_1)  #sum pooling
        global_agg_2 = self.global_aggregation_info( batch_2, mlp_source_2) 

        scores = torch.cat([global_agg_1, global_agg_2], dim=1)  # concatenation
        scores = self.mlp_last(scores)  # mlp  2*64-64 64-32 32-1

        return scores





class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x

        output_list = []
        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, 
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
            output_list.append(output)

        if self.norm is not None:
            output = self.norm(output)


        return output, output_list

class GraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool


    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
            
        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output, output_list = self.encoder(
            output, 
            edge_index, 
            complete_edge_index,
            edge_attr=edge_attr, 
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index, 
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, data.batch)

        return output, output_list

class SAT_GSL(nn.Module):
    def __init__(self, args):
        super(SAT_GSL, self).__init__()
        self.args = args
        self.num_node_features = self.args.num_features
        # self.gamma = torch.nn.Parameter(torch.Tensor(1))
        self.gamma = 0.01


        self.setup_layers()

    def setup_layers(self):
        self.feature_count = (self.args.output_dim + self.args.output_dim + self.args.output_dim) // 2

        self.sat = GraphTransformer(in_size=self.num_node_features,
                                 d_model=self.args.output_dim,
                                 dim_feedforward=2 * self.args.output_dim,
                                 dropout=self.args.dropout,
                                 num_heads=self.args.n_heads,
                                 num_layers=self.args.num_layers,
                                 batch_norm=self.args.batch_norm,
                                 abs_pe=self.args.abs_pe,
                                 abs_pe_dim=self.args.abs_pe_dim,
                                 gnn_type=self.args.gnn_type,
                                 k_hop=self.args.k_hop,
                                 use_edge_attr=False,
                                 num_edge_features=0,
                                 in_embed=False,
                                 se=self.args.se,
                                 use_global_pool=False)

        self.attention_pool = AttentionModule(self.args, self.args.output_dim)
        self.attention_level1 = AttentionModule(self.args, self.args.output_dim)
        self.attention_level2 = AttentionModule(self.args, self.args.output_dim)
        self.attention_level3 = AttentionModule(self.args, self.args.output_dim)

        self.tensor_network_level1 = TensorNetworkModule(self.args, dim_size=self.args.output_dim)
        self.tensor_network_level2 = TensorNetworkModule(self.args, dim_size=self.args.output_dim)
        self.tensor_network_level3 = TensorNetworkModule(self.args, dim_size=self.args.output_dim)

        self.score_attention = SEAttentionModule(self.args, self.feature_count)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)


        self.NTN = NTNNetwork(self.args, self.args.output_dim)

        self.mlp = nn.Sequential(
                        nn.Linear(self.args.tensor_neurons,  16),
                        nn.ReLU(),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Linear(8, 4),
                        nn.ReLU(),
                        nn.Linear(4, 1),
                        nn.Sigmoid()
                )


    def forward(self, data):
        # edge_index_1 = data["g1"].edge_index
        # edge_index_2 = data["g2"].edge_index
        # features_1 = data["g1"].x
        # features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        L_cl = 0
        if self.args.use_multiefn:
            features__1, feature1_list = self.sat(data["g1"])
            features__2, feature2_list = self.sat(data["g2"])

            pooled_features_level1_1 = self.attention_level1(feature1_list[0], batch_1)
            pooled_features_level1_2 = self.attention_level1(feature2_list[0], batch_2)
            scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)

            pooled_features_level2_1 = self.attention_level2(feature1_list[1], batch_1)
            pooled_features_level2_2 = self.attention_level2(feature2_list[1], batch_2)
            scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)

            pooled_features_level3_1 = self.attention_level3(feature1_list[2], batch_1)
            pooled_features_level3_2 = self.attention_level3(feature2_list[2], batch_2)
            scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)

            scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
            scores = F.relu(self.fully_connected_first(self.score_attention(scores) * scores + scores))
            score = torch.sigmoid(self.scoring_layer(scores)).view(-1)

            # L_cl = 0
            if self.args.use_LAreg:
                features_level1_1 = feature1_list[0]
                features_level2_1 = feature1_list[1]
                features_level3_1 = feature1_list[2]

                features_level1_2 = feature2_list[0]
                features_level2_2 = feature2_list[1]
                features_level3_2 = feature2_list[2]


                cat_node_embeddings_1 = torch.cat((features_level1_1, features_level2_1), dim=1)

                cat_node_embeddings_1 = torch.cat((cat_node_embeddings_1, features_level3_1), dim=1)

                cat_node_embeddings_2 = torch.cat((features_level1_2, features_level2_2), dim=1)
                cat_node_embeddings_2 = torch.cat((cat_node_embeddings_2, features_level3_2), dim=1)

                cat_global_embedding_1 = torch.cat((pooled_features_level1_1, pooled_features_level2_1), dim=1)
                cat_global_embedding_1 = torch.cat((cat_global_embedding_1, pooled_features_level3_1), dim=1)

                cat_global_embedding_2 = torch.cat((pooled_features_level1_2, pooled_features_level2_2), dim=1)
                cat_global_embedding_2 = torch.cat((cat_global_embedding_2, pooled_features_level3_2), dim=1)

                config_path = "LAreg_config/" + self.args.dataset + ".yml"
                self.config = get_config(config_path)
                self.config = self.config["GSC_GNN"]
                self.use_ssl = self.config.get('use_ssl', False)
                if self.use_ssl:
                    # self.GCL_model = GCL(self.config, sum(self.filters))
                    self.GCL_model = GCL(self.args, self.config, 1)
                    # print("self.gamma:", self.gamma.item())

                    if self.config['use_deepsets']:
                        L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2,
                                              g1=cat_global_embedding_1,
                                              g2=cat_global_embedding_2) * self.gamma  
                        # cat_node_embeddings_1 is the concatenation of the outputs of each layer of GNN,
                        # cat_global_embedding_1 is the concatenation of the pooled features output by each layer of GNN
                    else:
                        L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1,
                                              cat_node_embeddings_2) * self.gamma

                    if self.config.get('cl_loss_norm', False):
                        if self.config.get('norm_type', 'sigmoid') == 'sigmoid':
                            L_cl = torch.sigmoid(L_cl)
                        elif self.config.get('norm_type', 'sigmoid') == 'sum':
                            L_cl = torch.pow(L_cl, 2).sqrt()
                        elif self.config.get('norm_type', 'sigmoid') == 'tanh':
                            L_cl = torch.tanh(L_cl)
                        else:
                            raise "Norm Error"

        else:
            features__1, feature1_list = self.sat(data["g1"])
            features__2, feature2_list = self.sat(data["g2"])

            pooled_features__1 = self.attention_pool(features__1, batch_1)

            pooled_features__2 = self.attention_pool(features__2, batch_2)

            combinedFeature = self.NTN(pooled_features__1, pooled_features__2)

            score = self.mlp(combinedFeature).view(-1)

        return score, L_cl

class GCL(torch.nn.Module):

    def __init__(self, args, config, embedding_dim):
        super(GCL, self).__init__()
        self.args = args
        self.config = config
        self.use_deepsets = config['use_deepsets']
        self.use_ff = config['use_ff']
        self.embedding_dim = embedding_dim
        self.measure = config['measure']
        if self.use_ff:
            self.local_d = FF(self.embedding_dim)
            self.global_d = FF(self.embedding_dim)
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch_1, batch_2, z1, z2, g1=None, g2=None):

        if not self.use_deepsets:
            g1 = global_add_pool(z1, batch_1)
            g2 = global_add_pool(z2, batch_2)

        num_graphs_1 = g1.shape[0]
        num_nodes_1 = z1.shape[0]
        pos_mask_1 = torch.zeros((num_nodes_1, num_graphs_1)).to(self.args.device)
        num_graphs_2 = g2.shape[0]
        num_nodes_2 = z2.shape[0]
        pos_mask_2 = torch.zeros((num_nodes_2, num_graphs_2)).to(self.args.device)
        for node_idx, graph_idices in enumerate(zip(batch_1, batch_2)):
            g_idx_1, g_idx_2 = graph_idices
            pos_mask_1[node_idx][g_idx_1] = 1.
            pos_mask_2[node_idx][g_idx_2] = 1.

        if self.config.get('norm', False):
            z1 = F.normalize(z1, dim=1)
            g1 = F.normalize(g1, dim=1)
            z2 = F.normalize(z2, dim=1)
            g2 = F.normalize(g2, dim=1)
        self_sim_1 = torch.mm(z1, g1.t()) * pos_mask_1
        self_sim_2 = torch.mm(z2, g2.t()) * pos_mask_2
        cross_sim_12 = torch.mm(z1, g2.t()) * pos_mask_1
        cross_sim_21 = torch.mm(z2, g1.t()) * pos_mask_2
        # get_positive_expectation(self_sim_1,  self.measure, average=False)

        if self.config['sep']:
            self_js_sim_11 = get_positive_expectation(self_sim_1, self.measure, average=False).sum(1)
            cross_js_sim_12 = get_positive_expectation(cross_sim_12, self.measure, average=False).sum(1)

            self_js_sim_22 = get_positive_expectation(self_sim_2, self.measure, average=False).sum(1)
            cross_js_sim_21 = get_positive_expectation(cross_sim_21, self.measure, average=False).sum(1)
            L_1 = (self_js_sim_11 - cross_js_sim_12).pow(2).sum().sqrt()
            L_2 = (self_js_sim_22 - cross_js_sim_21).pow(2).sum().sqrt()
            return L_1 + L_2
        else:
            L_1 = get_positive_expectation(self_sim_1, self.measure, average=False).sum() - get_positive_expectation(
                cross_sim_12, self.measure, average=False).sum()
            L_2 = get_positive_expectation(self_sim_2, self.measure, average=False).sum() - get_positive_expectation(
                cross_sim_21, self.measure, average=False).sum()
            return L_1 - L_2

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.    
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep

def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))
