import pandas as pd
import dgl
import torch
from functools import partial
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from utils.featurizers import CanonicalAtomFeaturizer
from utils.featurizers import CanonicalBondFeaturizer
from dgllife.utils import one_hot_encoding
from scipy.sparse import coo_matrix
from utils.mcts import MCTS, reward_func
from utils.configures_shap import mcts_args, reward_args
from utils.shapley import GnnNets_GC2value_func
from dgllife.data.csv_dataset import MoleculeCSVDataset
import time
from rdkit import Chem
from sklearn.metrics import accuracy_score, recall_score, f1_score

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
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=args['self_loop']),
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=args['data_path'] +str(args['task'])+ '_graph.bin',
                                 task_names=[args['task_names']],
                                 load=False,init_mask=True,n_jobs=8
                            )

    return dataset

def find_closest_node_result(results, max_nodes):
    results = sorted(results, key=lambda x: x.P, reverse=True)
    results = sorted(results, key=lambda x: len(x.coalition))
    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node

def eval_subgraphx(args):
    df_test = pd.read_csv(args['csv_path_test'])
    test_set= load_data(args, df_test)
    test_loader = DataLoader(dataset=test_set,batch_size=1,collate_fn=collate_molgraphs)
    model = args['model']
    fn = args['model_path']
    model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
    model = model.to(device)
    fidelity_score_list = []
    infidelity_score_list = []
    sparsity_score_list = []
    accuracy_score_list = []
    recall_score_list = []
    true_aros = []
    pred_aros = []
    time_start = time.time()
    model.eval()
    index_to_explain = range(0, len(test_set))
    for iter, batch_data in enumerate(test_loader):
        if iter in index_to_explain:
            smiles, bg, label, masks = batch_data
            bg = bg.to(device)
            n_feats = bg.ndata['hv'].to(device)
            pred = model.forward(bg, n_feats).to(device)
            value_func = GnnNets_GC2value_func(model)
            mol = Chem.MolFromSmiles(smiles[0])
            adj_arr = Chem.GetAdjacencyMatrix(mol)
            coo_A = coo_matrix(adj_arr)
            b = [coo_A.row.tolist(), coo_A.col.tolist()]
            edges_index = torch.tensor(b).to(device)
            payoff_func = reward_func(reward_args, value_func)
            mcts_state_map = MCTS(batch_data, edges_index,
                                  score_func=payoff_func,
                                  n_rollout=mcts_args.rollout,
                                  min_atoms=mcts_args.min_atoms,
                                  c_puct=mcts_args.c_puct,
                                  expand_atoms=mcts_args.expand_atoms
                                  )
            results = mcts_state_map.mcts(verbose=True)
            graph_node_x = find_closest_node_result(results, max_nodes=50)
            masked_node_list = [node for node in list(range(graph_node_x.ori_graph.ndata['hv'].shape[0]))
                                if node not in graph_node_x.coalition]
            atoms = np.arange(n_feats.shape[0]).tolist()
            nodes = []
            for n in atoms:
                if n in masked_node_list:
                    nodes.append(0)
                else:
                    nodes.append(1)
            pred_aros.extend(nodes)
            unimportant_index = []
            for i, value in enumerate(nodes):
                if value == 0:
                    unimportant_index.append(i)
            b = torch.zeros(size=(n_feats.shape)).to(device)
            b[unimportant_index] = 1.0
            n_feats_masks = n_feats * b
            pred_unimp = model.forward(bg, n_feats_masks, get_node_gradient=False)
            fidelity_score = pred - pred_unimp
            fidelity_score = fidelity_score.abs().detach().cpu().numpy().flatten().tolist()
            fidelity_score_list.extend(fidelity_score)
            important_index = []
            for i, value in enumerate(nodes):
                if value == 1:
                    important_index.append(i)
            c = torch.zeros(size=(n_feats.shape)).to(device)
            c[important_index] = 1.0
            n_feats_masks_un = n_feats * c
            pred_imp = model.forward(bg, n_feats_masks_un, get_node_gradient=False)
            infidelity_score = pred - pred_imp
            infidelity_score = infidelity_score.abs().detach().cpu().numpy().flatten().tolist()
            infidelity_score_list.extend(infidelity_score)
            m = Chem.MolFromSmiles(smiles[0])
            aros = []
            for atom_idx in range(0, m.GetNumAtoms()):
                aro = m.GetAtomWithIdx(atom_idx).GetIsAromatic()
                if aro is False:
                    aros.append(0)
                else:
                    aros.append(1)
            sparsity_score = len(important_index) / len(aros)
            sparsity_score_list.append(sparsity_score)
            accuracy = accuracy_score(aros, nodes)
            accuracy_score_list.append(accuracy)
            recall = recall_score(aros, nodes)
            recall_score_list.append(recall)
            true_aros.extend(aros)
    time_end = time.time()
    print('Accuracy: ', accuracy_score(true_aros, pred_aros))
    print('Recall: ', recall_score(true_aros, pred_aros))
    print('F1: ', f1_score(true_aros, pred_aros))
    print('fidelity: ', sum(fidelity_score_list)/len(test_set))
    print('infidelity: ', sum(infidelity_score_list)/len(test_set))
    print('sparsity: ', sum(sparsity_score_list)/len(test_set))
    print('time cost', (time_end - time_start)/len(test_set))


if __name__ == '__main__':
    import argparse
    from configure_attribution import attribution_params

    parser = argparse.ArgumentParser(description='IG Attributions')
    parser.add_argument('-m', '--model', default='GAT',
                        choices=['GAT', 'GCN','Graphsage'])
    args = parser.parse_args().__dict__

    args.update(attribution_params(args))

    eval_subgraphx(args)