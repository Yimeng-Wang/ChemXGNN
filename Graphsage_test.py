import pandas as pd
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from utils.eval_meter import Meter
from utils.featurizers import CanonicalAtomFeaturizer
from utils.featurizers import CanonicalBondFeaturizer
from dgllife.utils import one_hot_encoding
from utils.graphsage_predictor import GraphSAGEPredictor
from dgllife.data.csv_dataset import MoleculeCSVDataset
from functools import partial


if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

import os
import random
import numpy as np

def set_random_seed(args):
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


def load_data(args,data):
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he',self_loop=args['self_loop'])
    n_feats = atom_featurizer.feat_size('hv')
    e_feats = bond_featurizer.feat_size('he')
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=args['self_loop']),
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=args['data_path'] +str(args['task'])+ '_graph.bin',
                                 task_names=[args['task_names']],
                                 load=False,init_mask=True,n_jobs=8
                            )

    return dataset,n_feats,e_feats

def eval_test(args, model, data_loader):
    model.eval()
    meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, label, masks = batch_data
        bg = bg.to(device)
        label = label.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        pred = model(bg, n_feats)
        meter.update(pred, label, masks)
    R2 = meter.compute_metric(args['metric'])
    print('R2:', R2)
    MAE = meter.compute_metric('mae')
    print('MAE:', MAE)
    RMSE = meter.compute_metric('rmse')
    print('RMSE:', RMSE)

def test(args):
    df_test = pd.read_csv(args['csv_path_test'])
    test_set,n_feats,e_feats = load_data(args, df_test)
    test_loader = DataLoader(dataset=test_set,
                              batch_size=128,
                              collate_fn=collate_molgraphs)

    model = GraphSAGEPredictor(in_feats=n_feats,
                               hidden_feats=[args['gnn_hidden_feats']] * args['num_layers'],
                               activation=[nn.LeakyReLU] * args['num_layers'],
                               aggregator_type=[args['aggregator']] * args['num_layers'],
                               predictor_hidden_feats=args['predictor_hidden_feats'],
                               n_tasks=1
                               )
    fn = args['model_path']
    model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
    gcn_net = model.to(device)

    eval_test(args, gcn_net, test_loader)


if __name__ == '__main__':
    import argparse
    from configure_model import graphsage_configure

    parser = argparse.ArgumentParser(description='GAT Prediction')
    parser.add_argument('-t', '--task', default=None,
                        choices=['Aromaticity', 'Lipophilicity'])
    args = parser.parse_args().__dict__

    args.update(graphsage_configure(args))

    test(args)