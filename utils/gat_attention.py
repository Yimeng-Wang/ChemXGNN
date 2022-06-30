import torch.nn as nn
import torch.nn.functional as F
from utils.gatconv import GATConv


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None, bias=True):
        super(GATLayer, self).__init__()

        self.gat_conv = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,allow_zero_in_degree=True,
                                negative_slope=alpha, residual=residual, bias=bias)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gat_conv.reset_parameters()

    def forward(self, bg, feats):
        feats,node_weights = self.gat_conv(bg, feats, get_attention=True)
        if self.agg_mode == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats,node_weights


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None):
        super(GAT, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0. for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(num_heads), len(feat_drops), len(attn_drops),
                   len(alphas), len(residuals), len(agg_modes), len(activations), len(biases)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, num_heads, ' \
                                       'feat_drops, attn_drops, alphas, residuals, ' \
                                       'agg_modes, activations, and biases to be the same, ' \
                                       'got {}'.format(lengths)
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GATLayer(in_feats, hidden_feats[i], num_heads[i],
                                            feat_drops[i], attn_drops[i], alphas[i],
                                            residuals[i], agg_modes[i], activations[i], 
                                            biases[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats,node_weights = gnn(g, feats)
        return feats,node_weights
