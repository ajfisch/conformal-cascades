"""Adapted from the chemprop repository."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chemprop


class MoleculeEncoder(nn.Module):
    """Neural network for encoding a molecule."""

    def __init__(self, hparams):
        super(MoleculeEncoder, self).__init__()
        self.hparams = hparams
        first_linear_dim = 0
        if hparams.use_mpn_features:
            self.mpn = MPN(hparams)
            first_linear_dim += hparams.mpn_hidden_size
        if hparams.use_mol_features:
            first_linear_dim += hparams.mol_features_size
        if not first_linear_dim:
            raise RuntimeError("Input dimension is 0.")

        ffnn = [nn.Dropout(hparams.dropout),
                nn.Linear(first_linear_dim, hparams.ffnn_hidden_size),
                nn.ReLU()]
        for _ in range(hparams.num_ffnn_layers - 2):
            ffnn.extend([
                nn.Dropout(hparams.dropout),
                nn.Linear(hparams.ffnn_hidden_size, hparams.ffnn_hidden_size),
                nn.ReLU()])
        ffnn.extend([nn.Dropout(hparams.dropout),
                     nn.Linear(hparams.ffnn_hidden_size, hparams.enc_hidden_size),
                     nn.ReLU()])
        self.ffnn = nn.Sequential(*ffnn)
        chemprop.nn_utils.initialize_weights(self)

    def forward(self, mol_graph, mol_features):
        inputs = []
        if self.hparams.use_mpn_features:
            inputs.append(self.mpn(mol_graph))
        if self.hparams.use_mol_features:
            if len(mol_features.shape) == 1:
                mol_features = mol_features.view(1, mol_features.shape[0])
            inputs.append(mol_features)
        inputs = torch.cat(inputs, dim=1)
        outputs = self.ffnn(inputs)
        return outputs


class MPN(nn.Module):
    """Message passing neural network for encoding a molecule."""

    def __init__(self, hparams):
        super(MPN, self).__init__()
        self.atom_fdim = chemprop.features.get_atom_fdim()
        self.bond_fdim = chemprop.features.get_bond_fdim()
        self.ffnn_hidden_size = hparams.mpn_hidden_size
        self.depth = hparams.mpn_depth
        self.dropout = hparams.dropout
        self.undirected = hparams.undirected_mpn

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.ffnn_hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.ffnn_hidden_size, bias=False)

        w_h_input_size = self.ffnn_hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.ffnn_hidden_size, bias=False)

        self.W_o = nn.Linear(self.atom_fdim + self.ffnn_hidden_size, self.ffnn_hidden_size)

    def forward(self, mol_graph):
        """Encodes a batch of molecular graphs."""
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph

        # Input
        inputs = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = F.relu(inputs)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            # Bond focused MPN.
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            # num_atoms x max_num_bonds x hidden
            nei_a_message = chemprop.nn_utils.index_select_ND(message, a2b)

            # num_atoms x hidden
            a_message = nei_a_message.sum(dim=1)

            # num_bonds x hidden
            rev_message = message[b2revb]

            # num_bonds x hidden
            message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = F.relu(inputs + message)
            message = self.dropout_layer(message)

        # num_atoms x max_num_bonds x hidden
        nei_a_message = chemprop.nn_utils.index_select_ND(message, a2b)

        # num_atoms x hidden
        a_message = nei_a_message.sum(dim=1)

        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)

        # num_atoms x hidden
        atom_hiddens = F.relu(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        # num_molecules x hidden
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs
