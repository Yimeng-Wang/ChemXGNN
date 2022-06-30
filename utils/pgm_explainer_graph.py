import torch
import numpy as np
import pandas as pd
from pgmpy.estimators.CITests import chi_square

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

def n_hops_A(A, n_hops):
    # Compute the n-hops adjacency matrix
    adj = torch.tensor(A, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.numpy().astype(int)

class Graph_Explainer:
    def __init__(
        self,
        model,
        graph,
        num_layers = None,
        perturb_feature_list = None,
        perturb_mode = "mean", # mean, zero, max or uniform
        perturb_indicator = "diff", # diff or abs
        print_result = 1
    ):
        self.model = model
        self.model.eval()
        self.graph = graph
        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.print_result = print_result
        self.X_feat = graph.ndata['hv']
        self.E_feat = graph.edata['he']
    
    def perturb_features_on_node(self, feature_matrix, node_idx, random = 0):

        X_perturb = feature_matrix.copy()
        perturb_array = X_perturb[node_idx].copy()
        epsilon = 0.05*np.max(self.X_feat.cpu().detach().numpy(), axis = 0)
        seed = np.random.randint(2)
        
        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if i in self.perturb_feature_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = np.mean(feature_matrix[:,i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = np.max(feature_matrix[:,i])
                        elif self.perturb_mode == "uniform":
                            perturb_array[i] = perturb_array[i] + np.random.uniform(low=-epsilon[i], high=epsilon[i])
                            if perturb_array[i] < 0:
                                perturb_array[i] = 0
                            elif perturb_array[i] > np.max(self.X_feat, axis = 0)[i]:
                                perturb_array[i] = np.max(self.X_feat, axis = 0)[i]

        
        X_perturb[node_idx] = perturb_array

        return X_perturb 
    
    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                            percentage, p_threshold, pred_threshold):
        X_torch = torch.tensor(self.X_feat.clone().detach().requires_grad_(True),
                               dtype=torch.float)
        pred_torch = self.model.forward(self.graph, X_torch)
        num_nodes = self.X_feat.shape[0]
        Samples = []
        for iteration in range(num_samples):
            X_perturb = self.X_feat.cpu().detach().numpy().copy()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb,
                                                                  node, random = latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)
            
            X_perturb_torch =  torch.tensor(X_perturb, dtype=torch.float).to(device)
            pred_perturb_torch = self.model.forward(self.graph, X_perturb_torch)
            pred_change = pred_torch.detach().cpu().numpy() - pred_perturb_torch.detach().cpu().numpy()
            sample.append(pred_change)
            Samples.append(sample)

        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)

        top = int(num_samples/8)
        top_idx = np.argsort(Samples[:,num_nodes])[-top:] 
        for i in range(num_samples):
            if i in top_idx:
                Samples[i,num_nodes] = 1
            else:
                Samples[i,num_nodes] = 0
            
        return Samples
    
    def explain(self, num_samples = 10, percentage = 50, top_node = None, p_threshold = 0.05,
                pred_threshold = 0.1):

        num_nodes = self.X_feat.shape[0]
        if top_node == None:
            top_node = int(num_nodes/20)
        
#         Round 1
        Samples = self.batch_perturb_features_on_node(int(num_samples/2), range(num_nodes),
                                                      percentage, p_threshold, pred_threshold)
        
        data = pd.DataFrame(Samples)
        
        p_values = []
        dependent_nodes1 = []
        
        target = num_nodes # The entry for the graph classification data is at "num_nodes"

        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
            if p <= p_threshold:
                dependent_nodes1.append(node)
        
        number_candidates = len(dependent_nodes1)
        try:
            candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]
        except:
            candidate_nodes = np.argpartition(p_values, number_candidates-1)[0:number_candidates-1]
#         Round 2
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage,
                                                      p_threshold, pred_threshold)
        data = pd.DataFrame(Samples)
        
        p_values2 = []
        dependent_nodes = []
        
        target = num_nodes
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values2.append(p)
            if p <= p_threshold:
                dependent_nodes.append(node)


        top_p = np.min((top_node,num_nodes-1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        return pgm_nodes, p_values2, candidate_nodes,num_nodes