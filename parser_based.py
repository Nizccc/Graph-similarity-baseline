
import argparse
from SAT_layers.position_encoding import POSENCODINGS
from SAT_layers.gnn_layers import GNN_TYPES


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run the code.")

    parser.add_argument('--seed',
                        type=int,
                        default=2024,
                        help='Random seed')
    
    parser.add_argument('--criterion',
                        type=str,
                        default='ged', #ged mcs eigenvalue degree
                        help='criterion')

    parser.add_argument('--log_path', type=str, default='./GEDLogs/', help='path for logfile') #'./MCSLogs/' './degreeLogs/'  './eigenvalueLogs/'

    parser.add_argument("--dataset",
                        nargs="?",
                        default="LINUX",
                        help="Dataset name. reg: AIDS700nef/LINUX/IMDBMulti")

    parser.add_argument('--cross_test',
                        type=str,
                        default='no', #yes no 
                        help='cross_test')
    
    parser.add_argument("--cross_dataset",
                        nargs="?",
                        default="IMDBMulti",
                        help="Cross Dataset name if cross_test. reg: LINUX/IMDBMulti")

    parser.add_argument("--epochs",
                        type=int,
                        default=10000,
                        help="Number of training epochs. Default is 10000.")
    
    parser.add_argument('--iter_val_start', 
                        type=int, 
                        default=3000) #3000

    parser.add_argument('--iter_val_every', 
                        type=int, 
                        default=3) #3

    parser.add_argument('--patience',
                        type=int,
                        default=100,
                        help='Patience for early stopping')
    

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="Dropout probability. Default is 0.0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")
    
    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Specify cuda devices')
    

#------------------
    parser.add_argument('--model',
                        type=str,
                        default='gnn_base', #gnn_base  mlp_base  SAT_GSL
                        help='which baseline')
    
    parser.add_argument("--nhid",
                        type=int,
                        default=64,
                        help="Hidden dimension in convolution. Default is 64.") 
       
#------------------  args for gnn_base  ------------------>
    parser.add_argument('--gnn_encoder',
                        type=str,
                        default='GIN', #GCN GAT GIN
                        help='which gnn for gnn_base')
       
    parser.add_argument('--gnn_filters',
                        type=list,
                        default=[64, 64, 64],
                        help='gnn_base layers') 
    
    parser.add_argument('--pooling',
                        type=str,
                        default='add', #add mean
                        help='gnn_base pooling')
    

#------------------  args for GraphTransformer (SAT_GSL)  ------------------>
    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="number of GraphTransformer layers. Default is 6.")

    parser.add_argument("--node_dim",
                        type=int,
                        default=64,
                        help="hidden dimensions of node features. Default is 128.")

    parser.add_argument("--edge_dim",
                        type=int,
                        default=4,
                        help="hidden dimensions of edge features. Default is 16.")

    parser.add_argument("--output_dim",
                        type=int,
                        default=64,
                        help="number of output node features dimensions. Default is 64.")

    parser.add_argument("--tensor_neurons",
                        type=int,
                        default=16,
                        help="number of Neurons for NTN. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--n_heads",
                        type=int,
                        default=2,
                        help="number of attention heads. Default is 4.")

    parser.add_argument("--synth",
                        dest="synth",
                        action="store_true")

    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')

    parser.add_argument('--batch_norm', type=bool, default=True)

    parser.add_argument('--abs-pe', type=str, default="rw", choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')

    parser.add_argument('--abs-pe-dim', type=int, default=3, help='dimension for absolute PE')

    parser.add_argument('--gnn-type', type=str, default='graph',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")

    parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")

    parser.add_argument('--se', type=str, default="gnn", help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--use_multiefn', type=bool, default=True)

    parser.add_argument('--use_LAreg', type=bool, default=False)

    parser.set_defaults(synth=False)

    if parser.parse_args().dataset in ["AIDS700nef", "LINUX"]:
        parser.set_defaults(max_in_degree=10)
        parser.set_defaults(max_out_degree=10)
        parser.set_defaults(max_path_distance=10)


    return parser.parse_args()

