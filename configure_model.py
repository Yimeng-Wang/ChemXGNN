import torch.nn as nn
import torch.nn.functional as F

def attentivefp_configure(args):
    if args['task'] == 'Aromaticity':
        attentivefp = {
            'task_names':'aromatic_count',
            'self_loop':False,
            'random_seed': 42,
            'graph_feat_size': 200,
            'num_layers': 2,
            'num_timesteps': 2,
            'predictor_hidden_feats':128,
            'dropout': 0.28,
            'weight_decay': 10 ** (-6.0),
            'lr': 0.001,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Aromaticity/aromaticity_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Aromaticity/aromaticity_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/AttentiveFP/aromaticity.pt'
        }
        return attentivefp
    if args['task'] == 'Lipophilicity':
        attentivefp = {
            'task_names':'Labels',
            'self_loop': False,
            'random_seed': 16,
            'graph_feat_size': 128,
            'num_layers': 2,
            'num_timesteps': 2,
            'predictor_hidden_feats':64,
            'dropout': 0.213,
            'weight_decay': 6.2558e-05,
            'lr': 0.000315,
            'batch_size': 128,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Lipophilicity/logP_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Lipophilicity/logP_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Lipophilicity/logP_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Lipophilicity/',
            'model_path': '../ChemXGNN/Models/AttentiveFP/lipophilicity.pt'
        }
        return attentivefp
    if args['task'] == 'Acute_oral_toxicity':
        attentivefp = {
            'task_names':'Labels',
            'self_loop': False,
            'random_seed': 6,
            'graph_feat_size': 64,
            'num_layers': 2,
            'num_timesteps': 1,
            'predictor_hidden_feats':256,
            'dropout': 0.26,
            'weight_decay': 4.1582e-05,
            'lr': 0.00095,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_test.csv',
            'patience': 25,
            'loss_fn': nn.BCEWithLogitsLoss(reduction='none'),
            'metric': 'roc_auc_score',
            'data_path': '../ChemXGNN/Data/Acute_oral_toxicity/',
            'model_path': '../ChemXGNN/Models/AttentiveFP/acute_oral_toxicity.pt'
        }

        return attentivefp

def gat_configure(args):
    if args['task'] == 'Aromaticity':
        gat = {
            'task_names':'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'gnn_hidden_feats': 256,
            'num_layers': 2,
            'num_heads': 4,
            'predictor_hidden_feats':128,
            'dropout': 0.245,
            'alphas': 0.9658,
            'residuals': True,
            'weight_decay': 7.9875e-05,
            'lr': 0.00115,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Aromaticity/aromaticity_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Aromaticity/aromaticity_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/GAT/aromaticity.pt'
        }
        return gat
    if args['task'] == 'Lipophilicity':
        gat = {
            'task_names':'Labels',
            'random_seed': 6,
            'self_loop': False,
            'gnn_hidden_feats': 128,
            'num_layers': 2,
            'num_heads': 8,
            'predictor_hidden_feats':64,
            'dropout': 0.214,
            'alphas': 0.333,
            'residuals': False,
            'weight_decay': 6.04027e-06,
            'lr': 0.00428,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Lipophilicity/logP_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Lipophilicity/logP_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Lipophilicity/logP_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Lipophilicity/',
            'model_path': '../ChemXGNN/Models/GAT/lipophilicity.pt'
        }
        return gat
    if args['task'] == 'Acute_oral_toxicity':
        gat = {
            'task_names':'Labels',
            'random_seed': 99,
            'self_loop': False,
            'gnn_hidden_feats': 64,
            'num_layers': 2,
            'num_heads': 4,
            'predictor_hidden_feats':256,
            'dropout': 0.2454,
            'alphas': 0.414,
            'residuals': True,
            'weight_decay': 6.0322e-05,
            'lr': 0.00461,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_test.csv',
            'patience': 30,
            'loss_fn': nn.BCEWithLogitsLoss(reduction='none'),
            'metric': 'roc_auc_score',
            'data_path': '../ChemXGNN/Data/Acute_oral_toxicity/',
            'model_path': '../ChemXGNN/Models/GAT/acute_oral_toxicity.pt'
        }

        return gat

def gcn_configure(args):
    if args['task'] == 'Aromaticity':
        gcn = {
            'task_names':'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'gnn_hidden_feats': 64,
            'num_layers': 2,
            'predictor_hidden_feats':64,
            'dropout': 0.05,
            'weight_decay': 3.7576e-06,
            'lr': 0.00361,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Aromaticity/aromaticity_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Aromaticity/aromaticity_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'patience': 25,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/GCN/aromaticity.pt'
        }
        return gcn
    if args['task'] == 'Lipophilicity':
        gcn = {
            'task_names':'Labels',
            'random_seed': 66,
            'self_loop': False,
            'gnn_hidden_feats': 256,
            'num_layers': 2,
            'predictor_hidden_feats':256,
            'dropout': 0.0566,
            'weight_decay': 2.672e-05,
            'lr': 0.000187,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Lipophilicity/logP_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Lipophilicity/logP_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Lipophilicity/logP_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Lipophilicity/',
            'model_path': '../ChemXGNN/Models/GCN/lipophilicity.pt'
        }
        return gcn
    if args['task'] == 'Acute_oral_toxicity':
        gcn = {
            'task_names':'Labels',
            'random_seed': 16,
            'self_loop': False,
            'gnn_hidden_feats': 64,
            'num_layers': 3,
            'predictor_hidden_feats':128,
            'dropout': 0.18,
            'weight_decay': 8.9057e-05,
            'lr': 0.01885,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_test.csv',
            'patience': 25,
            'loss_fn': nn.BCEWithLogitsLoss(reduction='none'),
            'metric': 'roc_auc_score',
            'data_path': '../ChemXGNN/Data/Acute_oral_toxicity/',
            'model_path': '../ChemXGNN/Models/GCN/acute_oral_toxicity.pt'
        }

        return gcn

def graphsage_configure(args):
    if args['task'] == 'Aromaticity':
        graphsage = {
            'task_names':'aromatic_count',
            'random_seed': 42,
            'self_loop': True,
            'gnn_hidden_feats': 128,
            'num_layers': 2,
            'aggregator': 'lstm',
            'predictor_hidden_feats':128,
            'dropout': 0.21,
            'weight_decay': 6.158e-05,
            'lr': 0.025,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Aromaticity/aromaticity_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Aromaticity/aromaticity_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Aromaticity/aromaticity_test.csv',
            'patience': 25,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Aromaticity/',
            'model_path': '../ChemXGNN/Models/Graphsage/aromaticity.pt'
        }
        return graphsage
    if args['task'] == 'Lipophilicity':
        graphsage = {
            'task_names':'Labels',
            'random_seed': 42,
            'self_loop': False,
            'gnn_hidden_feats': 128,
            'num_layers': 2,
            'aggregator': 'lstm',
            'predictor_hidden_feats':128,
            'dropout': 0.216,
            'weight_decay': 6.6154e-05,
            'lr': 0.0005067,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Lipophilicity/logP_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Lipophilicity/logP_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Lipophilicity/logP_test.csv',
            'patience': 30,
            'loss_fn': nn.MSELoss(reduction='none'),
            'metric': 'r2',
            'data_path': '../ChemXGNN/Data/Lipophilicity/',
            'model_path': '../ChemXGNN/Models/Graphsage/lipophilicity.pt'
        }
        return graphsage
    if args['task'] == 'Acute_oral_toxicity':
        graphsage = {
            'task_names':'Labels',
            'random_seed': 36,
            'self_loop': False,
            'gnn_hidden_feats': 128,
            'num_layers': 2,
            'aggregator': 'gcn',
            'predictor_hidden_feats':128,
            'dropout': 0.2547,
            'weight_decay': 9.33305e-07,
            'lr': 0.00657,
            'batch_size': 256,
            'num_epochs': 1000,
            'csv_path_train': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_train.csv',
            'csv_path_vali': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_valid.csv',
            'csv_path_test': '../ChemXGNN/Data/Acute_oral_toxicity/AoralT_test.csv',
            'patience': 30,
            'loss_fn': nn.BCEWithLogitsLoss(reduction='none'),
            'metric': 'roc_auc_score',
            'data_path': '../ChemXGNN/Data/Acute_oral_toxicity/',
            'model_path': '../ChemXGNN/Models/Graphsage/acute_oral_toxicity.pt'
        }

        return graphsage