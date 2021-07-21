import argparse
import torch
import ns.utils as utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict
import ns.modules as modules

torch.backends.cudnn.deterministic = True


def evaluate(args, args_eval, model_file):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset = utils.PathDatasetStateIds(
        hdf5_file=args.dataset, path_length=args_eval.num_steps)
    eval_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    # topk = [1, 5, 10]
    topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0

    pred_states = []
    next_states = []
    next_ids = []

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(eval_loader):
            data_batch = [[t.to(
                device) for t in tensor] for tensor in data_batch]
            observations, actions, state_ids = data_batch

            if observations[0].size(0) != args.batch_size:
                continue

            obs = observations[0]
            next_obs = observations[-1]
            next_id = state_ids[-1]

            state = model.obj_encoder(model.obj_extractor(obs))
            next_state = model.obj_encoder(model.obj_extractor(next_obs))

            pred_state = state
            for i in range(args_eval.num_steps):
                pred_trans = model.transition_model(pred_state, actions[i])
                pred_state = pred_state + pred_trans

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())
            next_ids.append(next_id.cpu().numpy())

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)
        next_ids_cat = np.concatenate(next_ids, axis=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = utils.pairwise_distance_matrix(
            next_state_flat, pred_state_flat)

        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)

        if args_eval.dedup:
            mask_mistakes = indices[:, 0] != 0
            closest_next_ids = next_ids_cat[indices[:, 0] - 1]

            if len(next_ids_cat.shape) == 2:
                equal_mask = np.all(closest_next_ids == next_ids_cat, axis=1)
            else:
                equal_mask = closest_next_ids == next_ids_cat

            indices[:, 0][np.logical_and(equal_mask, mask_mistakes)] = 0

        indices = torch.from_numpy(indices).long()

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum().item()

        pred_states = []
        next_states = []
        next_ids = []

    hits = hits_at[topk[0]] / float(num_samples)
    mrr = rr_sum / float(num_samples)

    print('Hits @ {}: {}'.format(topk[0], hits))
    print('MRR: {}'.format(mrr))

    return hits, mrr


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
