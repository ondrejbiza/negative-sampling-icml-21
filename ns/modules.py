import copy as cp
import ns.utils as utils

import numpy as np

import torch
from torch import nn


class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large',
                 ignore_action=False, copy_action=False, gamma=1.0, many_negs=False,
                 detach_negs=False, mix_negs=False):

        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.gamma = gamma
        self.many_negs = many_negs
        self.detach_negs = detach_negs
        self.mix_negs = mix_negs

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def energy_many_negs(self, state, neg_state):

        norm = 0.5 / (self.sigma ** 2)
        diff = state[:, None] - neg_state
        # sum over embedding dimension, average over num objects
        return norm * diff.pow(2).sum(3).mean(2)

    def energy_many_negs_mix(self, state, neg_state):

        norm = 0.5 / (self.sigma ** 2)
        diff = state[:, None, None] - neg_state[None, :, :]
        diff = diff.reshape((diff.shape[0], diff.shape[1] * diff.shape[2], diff.shape[3], diff.shape[4]))
        # sum over embedding dimension, average over num objects
        return norm * diff.pow(2).sum(3).mean(2)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs, custom_negs=None):

        assert not self.many_negs or custom_negs is not None

        state, next_state = self.extract_objects_(obs, next_obs)

        # Sample negative state across episodes at random
        neg_obs, neg_state = self.create_negatives_(obs, state)

        self.pos_loss = self.energy(state, action, next_state)
        self.pos_loss = self.pos_loss.mean()

        if custom_negs is not None:

            if self.many_negs:

                # [B, N, C, W, H]
                assert len(custom_negs.shape) == 5
                assert custom_negs.shape[0] == obs.shape[0]
                assert custom_negs.shape[2] == obs.shape[1]
                assert custom_negs.shape[3] == obs.shape[2]
                assert custom_negs.shape[4] == obs.shape[3]

                batch_size = custom_negs.shape[0]
                num_negs = custom_negs.shape[1]

                if self.detach_negs:

                    with torch.no_grad():

                        custom_neg_objs = self.obj_extractor(custom_negs.reshape(
                            (batch_size * num_negs, custom_negs.shape[2], custom_negs.shape[3], custom_negs.shape[4]))
                        )
                        custom_neg_state = self.obj_encoder(custom_neg_objs)
                        custom_neg_state = custom_neg_state.reshape(
                            (batch_size, num_negs, self.num_objects, self.embedding_dim)
                        )
                        custom_neg_state = custom_neg_state.detach()

                else:

                    custom_neg_objs = self.obj_extractor(custom_negs.reshape(
                        (batch_size * num_negs, custom_negs.shape[2], custom_negs.shape[3], custom_negs.shape[4]))
                    )
                    custom_neg_state = self.obj_encoder(custom_neg_objs)
                    custom_neg_state = custom_neg_state.reshape(
                        (batch_size, num_negs, self.num_objects, self.embedding_dim)
                    )

                self.negative_loss_many_(state, custom_neg_state)

            else:

                assert len(custom_negs.shape) == 4
                assert custom_negs.shape[0] == obs.shape[0]
                assert custom_negs.shape[1] == obs.shape[1]
                assert custom_negs.shape[2] == obs.shape[2]
                assert custom_negs.shape[3] == obs.shape[3]

                custom_neg_objs = self.obj_extractor(custom_negs)
                custom_neg_state = self.obj_encoder(custom_neg_objs)

                if self.detach_negs:
                    custom_neg_state = custom_neg_state.detach()

                self.negative_loss_(state, custom_neg_state)

        else:

            self.negative_loss_(state, neg_state)


        loss = self.pos_loss + self.gamma * self.neg_loss

        return loss

    def extract_objects_(self, obs, next_obs):

        state = self.forward(obs)
        next_state = self.forward(next_obs)

        return state, next_state

    def create_negatives_(self, obs, state):

        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_obs = obs[perm]
        neg_state = state[perm]

        return neg_obs, neg_state

    def negative_loss_(self, state, neg_state):

        self.neg_loss = self.hinge - self.energy(state, None, neg_state, no_trans=True)
        zeros = torch.zeros_like(self.neg_loss)
        self.neg_loss = torch.max(zeros, self.neg_loss)
        self.neg_loss = self.neg_loss.mean()

    def negative_loss_many_(self, state, neg_state):

        if self.mix_negs:
            energy = self.energy_many_negs_mix(state, neg_state)
            assert len(energy.shape) == 2
        else:
            energy = self.energy_many_negs(state, neg_state)
            assert len(energy.shape) == 2
            assert energy.shape[0] == neg_state.shape[0]
            assert energy.shape[1] == neg_state.shape[1]

        self.neg_loss = self.hinge - energy
        zeros = torch.zeros_like(self.neg_loss)
        self.neg_loss = torch.max(zeros, self.neg_loss)
        self.neg_loss = self.neg_loss.mean()

    def forward(self, obs):

        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu'):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_dim*2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        if num_objects > 1:
            node_input_dim = hidden_dim + self.input_dim + self.action_dim
        else:
            node_input_dim = self.input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, self.input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr, source_indices=None, target_indices=None):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr, source_indices=(row % self.num_objects).cpu().numpy(),
                target_indices=(col % self.num_objects).cpu().numpy())

        if not self.ignore_action:

            if self.copy_action:
                action_vec = utils.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = utils.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        return node_attr


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)
