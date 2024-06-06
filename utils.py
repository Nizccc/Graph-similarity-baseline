import torch
import math
import os
import json
import numpy as np
from texttable import Texttable
from torch.utils.data import random_split, Dataset
from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import GEDDataset
from ged_dataset import GEDDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, dense_to_sparse
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from torch_cluster import random_walk
from torch_sparse import spspmm, coalesce
import yaml
import torch_scatter
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform


def sort_edge_index(edge_index, edge_attr=None, num_nodes=None):
    r"""Row-wise sorts edge indices :obj:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    idx = edge_index[0] * num_nodes + edge_index[1]
    perm = idx.argsort()

    return edge_index[:, perm], None if edge_attr is None else edge_attr[perm]

class BinaryFuncDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.dir_name = os.path.join(root, name)
        self.number_features = 0
        self.func2graph = dict()
        super(BinaryFuncDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.func2graph, self.number_features = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        for filename in os.listdir(self.dir_name):
            if filename[-3:] == 'npy':
                continue
            f = open(self.dir_name + '/' + filename, 'r')
            contents = f.readlines()
            f.close()
            for jsline in contents:
                check_dict = dict()
                g = json.loads(jsline)
                funcname = g['fname']  # Type: str
                features = g['features']  # Type: list
                idlist = g['succs']  # Type: list
                n_num = g['n_num']  # Type: int
                # Build graph index
                edge_index = []
                for i in range(n_num):
                    idx = idlist[i]
                    if len(idx) == 0:
                        continue
                    for j in idx:
                        if (i, j) not in check_dict:
                            check_dict[(i,j)] = 1
                            edge_index.append((i,j))
                        if (j, i) not in check_dict:
                            check_dict[(j,i)] = 1
                            edge_index.append((j,i))
                np_edge_index = np.array(edge_index).T
                pt_edge_index = torch.from_numpy(np_edge_index)
                x = np.array(features, dtype=np.float32)
                x = torch.from_numpy(x)
                row, col = pt_edge_index
                cat_row_col = torch.cat((row,col))
                n_nodes = torch.unique(cat_row_col).size(0)
                if n_nodes != x.size(0):
                    continue
                self.number_features = x.size(1)
                pt_edge_index, _ = sort_edge_index(pt_edge_index, num_nodes=x.size(0))
                data = Data(x=x, edge_index=pt_edge_index)
                data.num_nodes = n_num
                if funcname in self.func2graph:
                    self.func2graph[funcname].append(data)
                else:
                    self.func2graph[funcname] = [data]
        torch.save((self.func2graph, self.number_features), self.processed_paths[0])

class FuncDataset(Dataset):
    def __init__(self, funcdict):
        super(FuncDataset, self).__init__()
        self.funcdict = funcdict
        self.id2key = dict()
        cnt = 0
        for k, v in self.funcdict.items():
            self.id2key[cnt] = k
            cnt += 1
    
    def __len__(self):
        return len(self.funcdict)

    def __getitem__(self, idx):
        graphset = self.funcdict[self.id2key[idx]]
        pos_idx = np.random.choice(range(len(graphset)), size=2, replace=True)
        origin_graph = graphset[pos_idx[0]]
        pos_graph = graphset[pos_idx[1]]
        all_keys = list(self.funcdict.keys())
        neg_key = np.random.choice(range(len(all_keys)))
        while all_keys[neg_key] == self.id2key[idx]:
            neg_key = np.random.choice(range(len(all_keys)))
        neg_data = self.funcdict[all_keys[neg_key]]
        neg_idx = np.random.choice(range(len(neg_data)))
        neg_graph = neg_data[neg_idx]
        
        return origin_graph, pos_graph, neg_graph, 1, 0

class GraphRegressionDataset(object):
    def __init__(self, args ,num, flag):
        self.args = args
        self.trainval_graphs = None
        # self.training_set = None
        self.val_set = None
        self.testing_graphs = None
        self.nged_matrix = None
        self.nmcs_matrix = None
        self.neigenvalue_matrix = None
        self.ndegree_matrix = None
        self.real_data_size = None
        self.number_features = None
        self.num=num
        self.flag=flag
        self.process_dataset()

    def process_dataset(self):
        if self.flag == 1:   #load cross_dataset
            print('\nPreparing cross_dataset.\n')
            self.trainval_graphs = GEDDataset('datasets/{}'.format(self.args.cross_dataset), self.args.cross_dataset, train=True)
            self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.cross_dataset), self.args.cross_dataset, train=False)
        else:
            print('\nPreparing dataset.\n')
            self.trainval_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)
            self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False)
        
        if self.args.criterion == 'ged':
            self.nged_matrix = self.trainval_graphs.norm_ged
            self.real_data_size = self.nged_matrix.size(0)
        elif self.args.criterion == 'mcs':
            self.nmcs_matrix = self.trainval_graphs.norm_mcs
            self.real_data_size = self.nmcs_matrix.size(0)
        elif self.args.criterion == 'eigenvalue':
            self.neigenvalue_matrix = self.trainval_graphs.norm_eigenvalue
            self.real_data_size = self.neigenvalue_matrix.size(0)        
        elif self.args.criterion == 'degree':
            self.ndegree_matrix = self.trainval_graphs.norm_degree
            self.real_data_size = self.ndegree_matrix.size(0)

        if self.args.dataset!='AIDS700nef':
            max_degree = 0
            for g in self.trainval_graphs + self.testing_graphs:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            if self.args.cross_test == 'yes': 
               max_degree = 88 
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.trainval_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.number_features = self.trainval_graphs.num_features

        # 5fold：
        val_ratio = 0.2
        num_trainval_gs = len(self.trainval_graphs)
        left = int(num_trainval_gs * (self.num-1) * val_ratio)
        right = int(num_trainval_gs * (self.num) * val_ratio)
        self.val_graphs = self.trainval_graphs[left:right]
        self.training_graphs = self.trainval_graphs[0:left] + self.trainval_graphs[right:]

    def create_batches(self, graphs):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        # """
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data['g1'] = data[0].to(self.args.device)
        new_data['g2'] = data[1].to(self.args.device)
        
        if self.args.criterion == 'ged':
            norm_ged = self.nged_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
            new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_ged])).view(-1).float().to(self.args.device)
        elif self.args.criterion == 'mcs':
            norm_mcs = self.nmcs_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
            new_data['target'] = torch.from_numpy(np.array([el for el in norm_mcs])).view(-1).float().to(self.args.device)
        if self.args.criterion == 'eigenvalue':
            norm_eigenvalue = self.neigenvalue_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
            new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_eigenvalue])).view(-1).float().to(self.args.device)
        if self.args.criterion == 'degree':
            norm_degree = self.ndegree_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
            new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_degree])).view(-1).float().to(self.args.device)
        
        return new_data

class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    t.add_rows([['Parameter', 'Value']] + [[k.replace('_', ' ').capitalize(), args[k]] for k in keys])
    print(t.draw())

def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]

def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps

def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
        fid = finds[0][0]
        while fid > 0:
            cid = sort_id_mat[fid]
            pid = sort_id_mat[fid - 1]
            if data[pid] == data[cid]:
                fid -= 1
            else:
                break
        rank[i] = fid + 1
    
    return rank

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation

def hypergraph_construction(edge_index, edge_attr, num_nodes, k=2, mode='RW'):
    if mode == 'RW':
        # Utilize random walk to construct hypergraph
        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walk = random_walk(row, col, start, walk_length=k)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
        adj[walk[start], start.unsqueeze(1)] = 1.0
        edge_index, _ = dense_to_sparse(adj)
    else:
        # Utilize neighborhood to construct hypergraph
        if k == 1:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
        else:
            neighbor_augment = TwoHopNeighbor()
            hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
            hop_data.num_nodes = num_nodes
            for _ in range(k-1):
                hop_data = neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    
    return edge_index, edge_attr

def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce='mean')

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce='sum')

    return hyperedge

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"

class OneHotDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features
    (functional name: :obj:`one_hot_degree`).

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(
        self,
        max_degree: int,
        in_degree: bool = False,
        cat: bool = True,
    ):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'



def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'min', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0

    return out



def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data

def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x

def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x
