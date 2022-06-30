"""
FileName: explainers.py
Description: Explainable methods' set
Time: 2020/8/4 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
from torch import Tensor
import torch.nn as nn
from utils.utils_GNN import subgraph
import dgl
from rdkit import Chem
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

EPS = 1e-15

from torch.nn import MSELoss
def loss_criterion(y_pred: torch.Tensor, y_true: torch.Tensor,):
    loss_fn= MSELoss(reduction='none')
    return loss_fn(y_pred, y_true)


class ExplainerBase(nn.Module):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=True, molecule=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self,x,init="normal"):

        (N, F)= x.size()
        self.node_feat_mask = torch.nn.Parameter(torch.randn(N, requires_grad=True, device=self.device) * 0.1)


    def __clear_masks__(self):
        self.node_feat_masks = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, node_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            item = item[subset]
            kwargs[key] = item

        return x, edge_index, mapping, node_mask, kwargs


    def forward(self,
                x: Tensor,
                bg,
                ex_labels,
                edge_index: Tensor,
                **kwargs
                ):

        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


    def control_sparsity(self, mask, sparsity=None):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7
        _, indices = torch.sort(mask, descending=True)
        mask_len = mask.shape[0]
        split_point = int((1 - sparsity) * mask_len)
        important_indices = indices[: split_point]
        unimportant_indices = indices[split_point:]
        trans_mask = mask.clone()
        trans_mask[important_indices] = float('inf')
        trans_mask[unimportant_indices] = - float('inf')

        return trans_mask

    def eval_related_pred(self, bg, x, node_feat_masks):
        node_idx = 0  # graph level: 0, node level: node_idx
        related_preds = []

        for ex_label, node_feat_mask in enumerate(node_feat_masks):

            self.node_feat_mask.data = float('inf') * torch.ones(node_feat_mask.size(), device=device)
            h = x * self.node_feat_mask.data.view(-1, 1).sigmoid()
            ori_pred = self.model(bg, h)

            self.node_feat_mask.data = node_feat_mask
            h = x * self.node_feat_mask.data.view(-1, 1).sigmoid()
            masked_pred = self.model(bg,h)

            # mask out important elements for fidelity calculation
            self.node_feat_mask.data = - node_feat_mask  # keep Parameter's id
            h = x * self.node_feat_mask.data.view(-1, 1).sigmoid()
            maskout_pred = self.model(bg,h)

            # zero_mask
            self.node_feat_mask.data = - float('inf') * torch.ones(node_feat_mask.size(),
                                                                   device=device)
            h = x * self.node_feat_mask.data.view(-1, 1).sigmoid()
            zero_mask_pred = self.model(bg,h)

            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

        return related_preds


class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    """

    coeffs = {
        'edge_size': 1.0,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, molecule=False):
        super(GNNExplainer, self).__init__(model, epochs, lr, molecule)



    def __loss__(self, raw_preds, x_label):
        loss = loss_criterion(raw_preds, x_label)
        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          bg,
                          ex_label: Tensor,
                          mask_features: True,
                          **kwargs
                          ) -> None:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask],lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(-1, 1).sigmoid()
            else:
                h = x
            raw_preds = self.model(bg,h, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        return self.node_feat_mask.data

    def forward(self, x, bg, ex_labels,mask_features=True,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        bg =dgl.add_self_loop(bg)

        # Calculate mask
        print('#D#Masks calculate...')
        node_feat_masks=[]
        for ex_label in ex_labels:
            self.__clear_masks__()
            self.__set_masks__(x)
            #edge_masks.append(self.control_sparsity(self.gnn_explainer_alg(x, edge_index, ex_label), sparsity=kwargs.get('sparsity')))
            node_feat_masks.append(self.gnn_explainer_alg(x,bg, ex_label,mask_features=True))


        print('#D#Predict...')

        with torch.no_grad():
            related_preds = self.eval_related_pred(bg,x, node_feat_masks, **kwargs)


        self.__clear_masks__()

        return node_feat_masks, related_preds




    def __repr__(self):
        return f'{self.__class__.__name__}()'


