import torch.nn as nn

class MLPPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.LayerNorm(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)
