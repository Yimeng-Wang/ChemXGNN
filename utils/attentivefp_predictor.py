import torch
import numpy as np
import torch.nn as nn
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


class AttentiveFPPredictor(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 predictor_hidden_feats=128,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, predictor_hidden_feats),
            nn.ReLU(),
            nn.LayerNorm(predictor_hidden_feats),
            nn.Linear(predictor_hidden_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False,
                get_node_gradient=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats,get_node_weight)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        if get_node_gradient:
            # Calculate graph representation by average readout.
            Final_feature = self.predict(graph_feats)
            baseline = torch.zeros(node_feats.shape).to(device)
            scaled_nodefeats = [baseline + (float(i) / 50) * (node_feats - baseline) for i in range(0, 51)]
            gradients = []
            for scaled_nodefeat in scaled_nodefeats:
                scaled_hg = self.readout(g, scaled_nodefeat)
                scaled_Final_feature = self.predict(scaled_hg)
                gradient = torch.autograd.grad(scaled_Final_feature[0][0], scaled_nodefeat)[0]
                gradient = gradient.detach().cpu().numpy()
                gradients.append(gradient)
            gradients = np.array(gradients)
            grads = (gradients[:-1] + gradients[1:]) / 2.0
            avg_grads = np.average(grads, axis=0)
            avg_grads = torch.from_numpy(avg_grads).to(device)
            integrated_gradients = (node_feats - baseline) * avg_grads
            phi0 = []
            for j in range(node_feats.shape[0]):
                a = sum(integrated_gradients[j].detach().cpu().numpy().tolist())
                phi0.append(a)
            node_gradient = torch.tensor(phi0)
            return Final_feature, node_gradient
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats)
