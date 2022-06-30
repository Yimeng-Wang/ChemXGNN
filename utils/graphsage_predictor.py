import numpy as np
import torch.nn as nn
import torch
from dgllife.model.gnn import GraphSAGE
from utils.mlp_predictor_grad import MLPPredictor
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

class GraphSAGEPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, activation=None, aggregator_type=None,
                 dropout=None, classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(GraphSAGEPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GraphSAGE(in_feats=in_feats,
                      hidden_feats=hidden_feats,
                      activation=activation,
                      aggregator_type=aggregator_type,
                      dropout=dropout
                             )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2*gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats,get_node_gradient=False):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        if get_node_gradient:
            # Calculate graph representation by average readout.
            Final_feature = self.predict(graph_feats)
            baseline = torch.zeros(node_feats.shape).to(device)
            scaled_nodefeats = [baseline + (float(i) / 50) * (node_feats - baseline) for i in range(0, 51)]
            gradients=[]
            for scaled_nodefeat in scaled_nodefeats:
                scaled_hg = self.readout(bg, scaled_nodefeat)
                scaled_Final_feature = self.predict(scaled_hg)
                gradient = torch.autograd.grad(scaled_Final_feature[0][0], scaled_nodefeat)[0]
                gradient=gradient.detach().cpu().numpy()
                gradients.append(gradient)
            gradients=np.array(gradients)
            grads = (gradients[:-1] + gradients[1:]) / 2.0
            avg_grads = np.average(grads, axis=0)
            avg_grads=torch.from_numpy(avg_grads).to(device)
            integrated_gradients = (node_feats - baseline) * avg_grads
            phi0 = []
            for j in range(node_feats.shape[0]):
                a = sum(integrated_gradients[j].detach().cpu().numpy().tolist())
                phi0.append(a)
            node_gradient = torch.tensor(phi0)
            return Final_feature,node_gradient
        else:
            Final_feature = self.predict(graph_feats)
            return Final_feature