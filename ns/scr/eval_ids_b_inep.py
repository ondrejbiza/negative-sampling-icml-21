import argparse
import torch
import ns.utils as utils
import os
import pickle

from torch.utils import data
import numpy as np
import ns.modules as modules

torch.backends.cudnn.deterministic = True


def evaluate(args, args_eval, model_file):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset = utils.PathDatasetStateIds(
        hdf5_file=args.dataset, path_length=10)
    eval_loader = data.DataLoader(
        dataset, batch_size=100, shuffle=False, num_workers=4)

    # Get data sample
    obs = eval_loader.__iter__().next()[0]
    input_shape = obs[0][0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    hits_list = []

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(eval_loader):

            data_batch = [[t.to(
                device) for t in tensor] for tensor in data_batch]

            observations, actions, state_ids = data_batch

            if observations[0].size(0) != args.batch_size:
                continue

            states = []
            for obs in observations:
                states.append(model.obj_encoder(model.obj_extractor(obs)))
            states = torch.stack(states, dim=0)
            state_ids = torch.stack(state_ids, dim=0)

            pred_state = states[0]
            if not args_eval.no_transition:
                for i in range(args_eval.num_steps):
                    pred_trans = model.transition_model(pred_state, actions[i])
                    pred_state = pred_state + pred_trans

            # pred_state: [100, |O|, D]
            # states: [10, 100, |O|, D]
            # pred_state_flat: [100, X]
            # states_flat: [10, 100, X]
            pred_state_flat = pred_state.reshape((pred_state.size(0), pred_state.size(1) * pred_state.size(2)))
            states_flat = states.reshape((states.size(0), states.size(1), states.size(2) * states.size(3)))

            # dist_matrix: [10, 100]
            dist_matrix = (states_flat - pred_state_flat[None]).pow(2).sum(2)
            indices = torch.argmin(dist_matrix, dim=0)
            correct = indices == args_eval.num_steps

            # check for duplicates
            if args_eval.dedup:
                equal_mask = torch.all(state_ids[indices, list(range(100))] == state_ids[args_eval.num_steps], dim=1)
                correct = correct + equal_mask

            # hits
            hits_list.append(correct.float().mean().item())

    hits = np.mean(hits_list)

    print('Hits @ 1: {}'.format(hits))

    return hits, 0.


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of prediction steps to evaluate.')
    parser.add_argument('--dataset', type=str,
                        default='data/shapes_eval.h5',
                        help='Dataset string.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--no-transition', default=False, action='store_true')
    parser.add_argument('--dedup', default=False, action='store_true')

    args_eval = parser.parse_args()

    meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
    model_file = os.path.join(args_eval.save_folder, 'model.pt')

    args = pickle.load(open(meta_file, 'rb'))['args']

    args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
    args.batch_size = 100
    args.dataset = args_eval.dataset
    args.seed = 0

    evaluate(args, args_eval, model_file)
