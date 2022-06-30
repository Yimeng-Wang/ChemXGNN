import pandas as pd
import dgl
import torch
from functools import partial
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping
from utils.eval_meter import Meter
from utils.featurizers import CanonicalAtomFeaturizer
from utils.featurizers import CanonicalBondFeaturizer
from dgllife.utils import one_hot_encoding
from utils.attentivefp_predictor import AttentiveFPPredictor
from dgllife.data.csv_dataset import MoleculeCSVDataset


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


def run_a_train_epoch(args,epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        batch_data
        smiles, bg, labels, masks = batch_data
        bg=bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric(args['metric']))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training {:.4f}, training_loss {:.4f}'.format(epoch + 1, args['num_epochs'], total_r2 ,total_loss))
    return total_r2, total_loss

def run_an_eval_epoch(args, model, data_loader,loss_criterion):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss=val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric(args['metric']))
        total_loss = np.mean(val_losses)
    return total_score, total_loss

def train(args):
    df_train = pd.read_csv(args['csv_path_train'])
    df_vali = pd.read_csv(args['csv_path_vali'])
    train_set,n_feats,e_feats = load_data(args, df_train)
    val_set,n_feats,e_feats = load_data(args, df_vali)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)

    model = AttentiveFPPredictor(node_feat_size=n_feats,
                                 edge_feat_size=e_feats,
                                 num_layers=args['num_layers'],
                                 num_timesteps=args['num_timesteps'],
                                 graph_feat_size=args['graph_feat_size'],
                                 predictor_hidden_feats=args['predictor_hidden_feats'],
                                 n_tasks=1,
                                 dropout=args['dropout']
                                 )
    model = model.to(device)
    loss_fn = args['loss_fn']
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    stopper = EarlyStopping(mode='higher', patience=args['patience'])

    for e in range(args['num_epochs']):
        run_a_train_epoch(args,e, model, train_loader, loss_fn, optimizer)
        val_score = run_an_eval_epoch(args, model, val_loader, loss_fn)
        early_stop = stopper.step(val_score[0], model)

        if e % 10 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
                e + 1, args['num_epochs'], args['metric'], val_score[0], 'loss', val_score[-1],
                args['metric'], stopper.best_score))
        if early_stop:
            break

    fn = args['model_path']
    torch.save(model.state_dict(), fn)


if __name__ == '__main__':
    import argparse
    from configure_model import attentivefp_configure

    parser = argparse.ArgumentParser(description='AttntiveFP Prediction')
    parser.add_argument('-t', '--task', default=None,
                        choices=['Aromaticity', 'Lipophilicity','Acute_oral_toxicity'])
    args = parser.parse_args().__dict__

    args.update(attentivefp_configure(args))

    train(args)