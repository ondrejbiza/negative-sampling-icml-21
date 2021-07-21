import argparse
import torch
import ns.utils as utils
import datetime
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import logging

from torch.utils import data

import ns.modules as modules


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--custom-neg', default=False, action='store_true')
parser.add_argument('--in-ep-prob', type=float, default=0.5)
parser.add_argument('--disable-time-aligned', default=False, action='store_true')
parser.add_argument('--num-negs', type=int, default=1)
parser.add_argument('--detach-negs', default=False, action='store_true')
parser.add_argument('--mix-negs', default=False, action='store_true')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--save-all', default=False, action='store_true')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.num_negs >= 1

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')
loss_file = os.path.join(save_folder, 'loss.pdf')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

if args.custom_neg:
    if args.num_negs == 1 and not args.mix_negs:
        dataset = utils.StateTransitionsDatasetNegs(
            hdf5_file=args.dataset, in_ep_prob=args.in_ep_prob, time_aligned=not args.disable_time_aligned
        )
    else:
        dataset = utils.StateTransitionsDatasetNegsMany(
            hdf5_file=args.dataset, in_ep_prob=args.in_ep_prob, time_aligned=not args.disable_time_aligned,
            num_negs=args.num_negs
        )
else:
    assert args.num_negs == 1
    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset
    )


def worker_init_fc(worker_index):
    # make sure each worker has a different starting random seed
    seed = np.random.get_state()[1][0]
    np.random.seed(seed + worker_index)


train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fc
)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

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
    gamma=args.gamma,
    encoder=args.encoder,
    many_negs=args.num_negs != 1 or args.mix_negs,
    detach_negs=args.detach_negs,
    mix_negs=args.mix_negs
).to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
losses = []

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0

    # make sure each epoch starts with a different random seed
    # we only need numpy random seed, we use it to make random decision in custom-neg workers
    np.random.seed(args.seed + epoch)

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        loss = model.contrastive_loss(*data_batch)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if args.save_all:
        torch.save(model.state_dict(), os.path.join(save_folder, "model_{:d}.pt".format(epoch)))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_file)

plt.subplot(2, 1, 1)
plt.plot(losses)
plt.subplot(2, 1, 2)
plt.plot(losses)
plt.yscale("log")
plt.savefig(loss_file)
