import torch.nn.functional as F
from utils.gcn_predictor import GCNPredictor

def attribution_params(args):
    if args['model'] == 'GCN':
        parameters = {
            'model': GCNPredictor(in_feats=78,
                             hidden_feats=[64, 64],
                             activation=[F.relu, F.relu],
                             n_tasks=1,
                             predictor_hidden_feats=64,
                             ),
            'task':'Aromaticity',
            'task_names': 'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/GCN/aromaticity.pt'
        }
        return parameters
    if args['model'] == 'GAT':
        from utils.gat_predictor import GATPredictor
        parameters = {
            'model': GATPredictor(in_feats=78,
                             hidden_feats=[256, 256],
                             num_heads=[4, 4],
                             alphas=[0.9658, 0.9658],
                             residuals=[True, True],
                             predictor_hidden_feats=128,
                             n_tasks=1
                             ),
            'task': 'Aromaticity',
            'task_names':'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/GAT/aromaticity.pt'
        }
        return parameters
    if args['model'] == 'Graphsage':
        from utils.graphsage_predictor import GraphSAGEPredictor
        import torch.nn as nn
        parameters = {
            'model': GraphSAGEPredictor(in_feats=78,
                                   hidden_feats=[128, 128],
                                   activation=[nn.LeakyReLU, nn.LeakyReLU],
                                   aggregator_type=['lstm', 'lstm'],
                                   predictor_hidden_feats=128,
                                   n_tasks=1
                                   ),
            'task': 'Aromaticity',
            'task_names':'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/Graphsage/aromaticity.pt'
        }
        return parameters
    if args['model'] == 'AttentiveFP':
        from utils.attentivefp_predictor import AttentiveFPPredictor
        parameters = {
            'model':AttentiveFPPredictor(node_feat_size=78,
                                     edge_feat_size=12,
                                     num_layers=2,
                                     num_timesteps=2,
                                     graph_feat_size=200,
                                     predictor_hidden_feats=128,
                                     n_tasks=1,
                                     ),
            'task': 'Aromaticity',
            'task_names':'aromatic_count',
            'self_loop':False,
            'random_seed': 42,
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/AttentiveFP/aromaticity.pt'
        }
        return parameters




