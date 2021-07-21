import argparse
import copy as cp
import gym
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from a3c.baby_a3c import NNPolicy, prepro
import a3c.utils as utils


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):

        self.action_space = action_space

    def act(self, observation, reward, done):

        del observation, reward, done
        return self.action_space.sample()


def init_env(env_name, seed):
    # initialize environment, get number of actions
    env = gym.make(env_name)
    env.seed(seed)
    num_actions = env.action_space.n  # get the action space of this game
    return env, num_actions


def init_and_load_model(hidden, num_actions, save_dir):
    # load a model and throw an error if model not found
    model = NNPolicy(channels=1, memsize=hidden, num_actions=num_actions)
    step = model.try_load(save_dir)
    assert step != 0
    return model


def init_random_agent(env):

    return RandomAgent(env.action_space)


def select_action(state, model, hx, eps):
    # select an action using either an epsilon greedy or softmax policy
    value, logit, hx = model((state.view(1, 1, 80, 80), hx))
    logp = F.log_softmax(logit, dim=-1)

    if eps is not None:
        # use epsilon greedy
        if np.random.uniform(0, 1) < eps:
            # random action
            return np.random.randint(logp.size(1))
        else:
            return torch.argmax(logp, dim=1).numpy()[0]
    else:
        # sample from softmax
        action = torch.exp(logp).multinomial(num_samples=1).data[0]
        return action.numpy()[0]


def reset_rnn_state():
    # reset the hidden state of an rnn
    return torch.zeros(1, 256)


def preprocess_state(state):

    return torch.tensor(prepro(state))


def crop_normalize(img, crop_ratio):

    img = cp.deepcopy(img)
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


def replay_init_episode(replay_buffer):

    replay_buffer.append({
        'obs': [],
        'action': [],
        'next_obs': [],
        'state_ids': [],
        'next_state_ids': []
    })


def construct_start_states_set(dup_paths):
    # go through a dataset and find all start states
    # we use this to reduce the overlap between train/valid/test sets
    blacklist_state_ids = set()

    for path in dup_paths:

        f = h5py.File(path, "r")

        # iterate over all episodes, stored in a dict
        for ep in f.values():
            # cheap way of making an immutable array
            blacklist_state_ids.add(ep['state_ids'][0].tobytes())

        f.close()

    return blacklist_state_ids


def main(args):
    # get arguments
    hidden = 256
    max_episodes = args.max_episodes
    max_steps = args.num_steps
    dataset_save_path = args.save_path
    min_burnin = args.min_burnin
    max_burnin = args.max_burnin
    env_name = args.env_id
    seed = args.seed
    save_dir = "./{:s}/".format(env_name.lower())

    torch.manual_seed(seed)

    # init environment
    env, num_actions = init_env(env_name, seed)
    model = init_and_load_model(hidden, num_actions, save_dir)
    random_agent = init_random_agent(env)

    episode_length, epr, eploss, done = 0, 0, 0, True
    state = env.reset()
    prev_state = state
    hx = reset_rnn_state()

    if env_name == 'PongDeterministic-v4':
        crop = (35, 190)
    elif env_name == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
    else:
        raise NotImplementedError("Only Pong and Space were used in the original paper.")

    replay_buffer = []

    # maybe get a blacklist of starting states
    blacklist_state_ids = None
    if args.check_dup_paths:
        blacklist_state_ids = construct_start_states_set(args.check_dup_paths)

    with torch.no_grad():
        # sample a random number of steps we take before collecting data
        burnin_steps = np.random.randint(min_burnin, max_burnin)
        replay_init_episode(replay_buffer)

        while True:

            episode_length += 1

            start_collection = episode_length > burnin_steps

            if start_collection:
                replay_buffer[-1]['state_ids'].append(
                    np.array(cp.deepcopy(env.unwrapped._get_ram()), dtype=np.int32)
                )

            if start_collection:
                # take random actions when collecting data
                action = random_agent.act(None, None, None)
            else:
                # otherwise use the trained agent
                action = select_action(preprocess_state(state), model, hx, args.eps)

            next_state, reward, done, _ = env.step(action)

            if env_name == 'PongDeterministic-v4':
                # reset when we win/lose a round (pos/neg reward)
                # don't reset once we are collecting random data
                # if we do reset, the dataset is extremely limited
                # because we only allow full 10-step episodes
                if reward != 0 and not start_collection:
                    done = True
            elif env_name == 'SpaceInvadersDeterministic-v4':
                # reset when we lose life (we start with 3 lives)
                if env.env.ale.lives() != 3:
                    done = True

            if blacklist_state_ids is not None:
                # first step of data collection
                if episode_length == burnin_steps + 1:
                    # if this start state exists in the training set, go to the next episode
                    if replay_buffer[-1]['state_ids'][-1].tobytes() in blacklist_state_ids:
                        print("duplicate start state, skip episode")
                        done = True

            if start_collection:
                state_replay = np.concatenate(
                    (crop_normalize(prev_state, crop), crop_normalize(state, crop)), axis=0
                )
                next_state_replay = np.concatenate(
                    (crop_normalize(state, crop), crop_normalize(next_state, crop)), axis=0
                )
                replay_buffer[-1]['obs'].append(state_replay)
                replay_buffer[-1]['next_obs'].append(next_state_replay)
                replay_buffer[-1]['action'].append(action)
                replay_buffer[-1]['next_state_ids'].append(
                    np.array(cp.deepcopy(env.unwrapped._get_ram()), dtype=np.int32)
                )

            epr += reward
            done = done or episode_length >= 1e4

            prev_state = state
            state = next_state

            num_samples = len(replay_buffer[-1]['obs'])
            if num_samples == max_steps:
                done = True

            if done:
                print("ep {:d}, length: {:d}".format(len(replay_buffer), episode_length))

                hx = reset_rnn_state()
                episode_length, epr, eploss = 0, 0, 0
                state = env.reset()
                prev_state = state

                # check if episode was long enough
                if num_samples != max_steps:
                    del replay_buffer[-1]

                # termination condition
                if len(replay_buffer) == max_episodes:
                    break

                burnin_steps = np.random.randint(min_burnin, max_burnin)
                replay_init_episode(replay_buffer)

    env.close()
    utils.save_list_dict_h5py(replay_buffer, dataset_save_path)


parser = argparse.ArgumentParser("Collect data from an Atari games using a trained A3C agent.")
parser.add_argument("env_id")
parser.add_argument("--min-burnin", type=int, default=None)
parser.add_argument("--max-burnin", type=int, default=None)
parser.add_argument("--max-episodes", type=int, default=None)
parser.add_argument("--num-steps", type=int, default=None)
parser.add_argument("--save-path", default=None)
parser.add_argument("--check-dup-paths", nargs="+", default=None)
parser.add_argument("--eps", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)
parsed = parser.parse_args()
main(parsed)
