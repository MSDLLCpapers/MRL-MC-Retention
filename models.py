import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout



class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)

    def forward(self, g):
        node_feats, edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats



class GINLayerModified(nn.Module):
    def __init__(self, num_edge_emb, emb_dim, batch_norm=True, activation=None):
        super(GINLayerModified, self).__init__()
        self.edge_embeddings = nn.Linear(num_edge_emb, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.LayerNorm(2 * emb_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        self.edge_embeddings.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        edge_embeds = self.edge_embeddings(edge_feats)
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats




'''GIN model'''
# from dgllife.model.gnn.gin import GIN
class GINModel(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    This module is for updating node representations only.

    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 200.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.

        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0
    """
    def __init__(self, num_node_emb, num_edge_emb,
                 num_layers=5, emb_dim=200, JK='concat', dropout=0.1,gru_out_layer=2):
        super(GINModel, self).__init__()

        self.num_layers = num_layers
        self.JK = 'concat'
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.node_embeddings = nn.Linear(num_node_emb, emb_dim)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayerModified(num_edge_emb, emb_dim))
            else:
                self.gnn_layers.append(GINLayerModified(num_edge_emb, emb_dim, activation=F.leaky_relu))

        self.readout = AttentiveFPReadout(
            emb_dim, num_timesteps=gru_out_layer, dropout=dropout
        )
        self.out = nn.Sequential(
            nn.Linear(emb_dim*(num_layers+1), 500),
            nn.LayerNorm(500),
            nn.LeakyReLU(),
            nn.Linear(500,200),
            nn.LayerNorm(200),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(200,50),
            nn.LayerNorm(50),
            #nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 1)
        )
        self.reset_parameters()

    def swizzle(self):
        self.out = nn.Sequential(
            nn.Linear(200*(4+1), 500),
            nn.LayerNorm(500),
            nn.LeakyReLU(),
            #nn.Linear(500,200),
            #nn.LayerNorm(200),
            #nn.LeakyReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(200,50),
            #nn.LayerNorm(50),
            #nn.BatchNorm1d(50),
            #nn.LeakyReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(50, 1)
        )


    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_embeddings.reset_parameters()
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g):
        """Update node representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(self.node_embeddings)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size. In particular, M will be emb_dim * (num_layers + 1)
            if self.JK == 'concat' and emb_dim otherwise.
        """
        categorical_node_feats, categorical_edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]

        node_embeds = self.node_embeddings(categorical_node_feats)

        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[layer],
                                                categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        feats = self.readout(g, final_node_feats)
        feats = self.out(feats)
        return feats




if __name__ == "__main__":
    pass

