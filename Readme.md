# ChemXGNN

A benchmarking platform for deep interpretable methods based on chemical GNNs.

## Installation

The code in this repository relies on the DGL package (https://github.com/dmlc/dgl) and pytorch-geometric pakage (https://github.com/pyg-team/pytorch_geometric) with pytorch backend (http://pytorch.org) as well as on sklearn (http://scikit-learn.org) and dgllife (https://github.com/awslabs/dgl-lifesci).
We are recommended you to create a conda environment for example:

`conda env create -f environment.yml`  

Then activate the environment:  
  
`conda activate ChemXGNN`

## Usage

These are the codes generating explanations for the aromaticity task (demonstrated in the paper), which can be run to obtain evaluation results for each XAI method.

To get attention weitghts from AttentiveFP models run: `python AttWeights.py -m AttentiveFP`

To get attention weitghts from GAT models run: `python AttWeights_gat.py -m GAT`

To run IG run: `python IG.py -m [Model]` 

To run GNNExplainer run: `python GNN_explainer.py -m [Model]`
  
To run PGMExplainer run: `python PGM_explainer.py -m [Model]`

To run SubgraphX run: `python SubgraphX.py -m [Model]`

we include our notebook **visualize_aromaticity.ipynb** to demonstrate how to visualize the explainations. Besides this, we also provided the interpretations of each XAI method for lipophilicity and acute oral toxicity tasks along with the visualizations, available in **visualize_lipophilicity.ipynb** and **visualize_OralAcuteToxicity**.

## (Optional) Train your own models

It is recommended to work with our trained models, or you can retrain them. By default, the script will use datasets with preprocessed molecules, and save model checkpoint in the current working directory. 

Step one, train the models using the following command：
  
  * `python AttentiveFP_train.py -t [Task]`
  * `python GAT_train.py -t [Task]`
  * `python Graphsage_train.py -t [Task]`
  * `python GCN_train.py -t [Task]`

Step two, employ the trained models to make predictions:  
  
  * `python AttentiveFP_test.py -t [Task]`
  * `python GAT_test.py -t [Task]`
  * `python Graphsage_test.py -t [Task]`
  * `python GCN_test.py -t [Task]`

## Cite

If you use this code (or parts thereof), please use the following BibTeX entry:

` `