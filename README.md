# Negative sampling in C-SWM

This is the original source code for "The Impact of Negative Sampling on Contrastive Structured World Models. O Biza, E van der Pol, T Kipf. ICML'21 SSL workshop."

We build upon the original C-SWM source code (/ns, https://github.com/tkipf/c-swm, MIT license) and use an open-source implementation of A3C (/a3c, https://github.com/greydanus/baby-a3c, MIT license).

## Setup

Tested with Python 3.6 on Ubuntu. Install required packages using
```
pip install -r requirements.txt
```

# Usage

## Generate data with a random policy

Generate training and validation trajectories using a random policy.
Shapes and Cubes immovable are used in Section 4.1 in the paper. Pong and Space Invaders with random actions appear in Section 4.2.
The size of all (uncompressed) datasets is around 40 GB.

```
# Shapes and Cubes immovable
python -m ns.data_gen.env --env_id ShapesImmovableTrain-v0 --fname data/shapes_imm_train.h5 --num_episodes 1000 --seed 1
python -m ns.data_gen.env --env_id ShapesImmovableEval-v0 --fname data/shapes_imm_eval.h5 --num_episodes 10000 --seed 2 --save-state-ids

python -m ns.data_gen.env --env_id CubesImmovableTrain-v0 --fname data/cubes_imm_train.h5 --num_episodes 1000 --seed 3
python -m ns.data_gen.env --env_id CubesImmovableEval-v0 --fname data/cubes_imm_eval.h5 --num_episodes 10000 --seed 4 --save-state-ids

# Atari Pong and Space Invaders
python -m ns.data_gen.env --env_id PongDeterministic-v4 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
python -m ns.data_gen.env --env_id PongDeterministic-v4 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2 --save-state-ids

python -m ns.data_gen.env --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_train.h5 --num_episodes 1000 --atari --seed 1
python -m ns.data_gen.env --env_id SpaceInvadersDeterministic-v4 --fname data/spaceinvaders_eval.h5 --num_episodes 100 --atari --seed 2 --save-state-ids
```

## Generate Atari dataset with A3C

These datasets are used in Section 4.3 in the paper.

First train A3C on Pong and Space Invaders. A3C is trained on a CPU in parallel. This took around a day on a high-end CPU for each game.

```
python -m a3c.baby_a3c --env PongDeterministic-v4
python -m a3c.baby_a3c --env SpaceInvadersDeterministic-v4
```

Generate train/valid/test datasets and make sure the start states in these datasets do not overlap.
Around 50 GB of data.

```
python -m a3c.scr.collect PongDeterministic-v4 --min-burnin 58 --max-burnin 100 --max-episodes 10000 --num-steps 10 --save-path data/pong_full_train_eps_0_5.h5 --seed 1 --eps 0.5
python -m a3c.scr.collect PongDeterministic-v4 --min-burnin 58 --max-burnin 100 --max-episodes 1000 --num-steps 10 --save-path data/pong_full_valid_dup_eps_0_5.h5 --seed 2 --check-dup-paths data/pong_full_train_eps_0_5.h5 --eps 0.5
python -m a3c.scr.collect PongDeterministic-v4 --min-burnin 58 --max-burnin 100 --max-episodes 1000 --num-steps 10 --save-path data/pong_full_test_dup_eps_0_5.h5 --seed 3 --check-dup-paths data/pong_full_train_eps_0_5.h5 data/pong_full_valid_dup_eps_0_5.h5 --eps 0.5

python -m a3c.scr.collect SpaceInvadersDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 10000 --num-steps 10 --save-path data/spaceinvaders_full_train_eps_0_5.h5 --seed 1 --eps 0.5
python -m a3c.scr.collect SpaceInvadersDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 1000 --num-steps 10 --save-path data/spaceinvaders_full_valid_dup_eps_0_5.h5 --seed 2 --check-dup-paths data/spaceinvaders_full_train.h5 --eps 0.5
python -m a3c.scr.collect SpaceInvadersDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 1000 --num-steps 10 --save-path data/spaceinvaders_full_test_dup_eps_0_5.h5 --seed 3 --check-dup-paths data/spaceinvaders_full_train.h5 data/spaceinvaders_full_valid_dup.h5 --eps 0.5
```

## Train models

Train and evaluate C-SWMs with baseline and our negative sampling strategies.

Shapes and Cubes Immovable: baseline vs episodic and out-of-episode negatives (Section 4.1).

```
# baseline shapes and cubes immovable
python -m ns.scr.train --dataset data/shapes_imm_train.h5 --encoder small --name shapes_imm
python -m ns.scr.eval_ids --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/shapes_imm --num-steps 1 --dedup

python -m ns.scr.train --dataset data/cubes_imm_train.h5 --encoder large --name cubes_imm
python -m ns.scr.eval_ids --dataset data/cubes_imm_eval.h5 --save-folder checkpoints/cubes_imm --num-steps 1 --dedup

# episodic and out-of-episode negatives for shapes and cubes immovable (beta=0.5)
python -m ns.scr.train --dataset data/shapes_imm_train.h5 --encoder small --name shapes_imm_neg --custom-neg --disable-time-aligned
python -m ns.scr.eval_ids --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/shapes_imm_neg --num-steps 1 --dedup

python -m ns.scr.train --dataset data/cubes_imm_train.h5 --encoder large --name cubes_imm_neg --custom-neg --disable-time-aligned
python -m ns.scr.eval_ids --dataset data/cubes_imm_eval.h5 --save-folder checkpoints/cubes_imm_neg --num-steps 1 --dedup
```

Pong and Space Invaders: baseline vs time-aligned negatives (Section 4.2).

```
# baseline pong and space invaders
python -m ns.scr.train --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong
python -m ns.scr.eval_ids --dataset data/pong_eval.h5 --save-folder checkpoints/pong --num-steps 1

python -m ns.scr.train --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 5 --copy-action --epochs 200 --name spaceinvaders
python -m ns.scr.eval_ids --dataset data/spaceinvaders_eval.h5 --save-folder checkpoints/spaceinvaders --num-steps 1

# time-aligned negatives
python -m ns.scr.train --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong_neg --custom-neg --in-ep-prob 0.0
python -m ns.scr.eval_ids --dataset data/pong_eval.h5 --save-folder checkpoints/pong_neg --num-steps 1

python -m ns.scr.train --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 5 --copy-action --epochs 200 --name spaceinvaders_neg --custom-neg --in-ep-prob 0.0
python -m ns.scr.eval_ids --dataset data/spaceinvaders_eval.h5 --save-folder checkpoints/spaceinvaders_neg --num-steps 1
```

Full Pong and Space Invaders: baseline vs episodic and out-of-episode negatives under "local" and "global" evaluation (Section 4.3).

```
# baseline
python -m ns.scr.train --dataset data/pong_full_train_eps_0_5.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 100 --name pong_full --learning-rate 0.0005
python -m ns.scr.eval_ids_b --dataset data/pong_full_test_dup_eps_0_5.h5 --save-folder checkpoints/pong_full --num-steps 1 --dedup # global eval
python -m ns.scr.eval_ids_b_inep --dataset data/pong_full_test_dup_eps_0_5.h5 --save-folder checkpoints/pong_full --num-steps 1 --dedup # local eval

python -m ns.scr.train --dataset data/spaceinvaders_full_train_eps_0_5.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 5 --copy-action --epochs 100 --name spaceinvaders_full --learning-rate 0.0005
python -m ns.scr.eval_ids_b --dataset data/spaceinvaders_full_test_dup_eps_0_5.h5 --save-folder checkpoints/spaceinvaders_full --num-steps 1 --dedup # global eval
python -m ns.scr.eval_ids_b_inep --dataset data/spaceinvaders_full_test_dup_eps_0_5.h5 --save-folder checkpoints/spaceinvaders_full --num-steps 1 --dedup # local eval

# episodic and out-of-episode negatives (beta=0.5)
python -m ns.scr.train --dataset data/pong_full_train_eps_0_5.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 100 --name pong_full_neg --learning-rate 0.0005 --custom-neg --disable-time-aligned
python -m ns.scr.eval_ids_b --dataset data/pong_full_test_dup_eps_0_5.h5 --save-folder checkpoints/pong_full_neg --num-steps 1 --dedup # global eval
python -m ns.scr.eval_ids_b_inep --dataset data/pong_full_test_dup_eps_0_5.h5 --save-folder checkpoints/pong_full_neg --num-steps 1 --dedup # local eval

python -m ns.scr.train --dataset data/spaceinvaders_full_train_eps_0_5.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 5 --copy-action --epochs 100 --name spaceinvaders_full_neg --learning-rate 0.0005 --custom-neg --disable-time-aligned
python -m ns.scr.eval_ids_b --dataset data/spaceinvaders_full_test_dup_eps_0_5.h5 --save-folder checkpoints/spaceinvaders_full_neg --num-steps 1 --dedup # global eval
python -m ns.scr.eval_ids_b_inep --dataset data/spaceinvaders_full_test_dup_eps_0_5.h5 --save-folder checkpoints/spaceinvaders_full_neg --num-steps 1 --dedup # local eval
```

# Code structure

* Environments are implemented in */ns/envs* and registered in */ns/envs/\_\_init\_\_.py* using OpenAI gym.
* The training script for A3C is in */a3c/baby_a3c.py*. Parts of this script are re-used in */a3c/scr/collect.py*.
* C-SWM is implemented in */ns/modules.py* and trained using */ns/scr/train.py*.
* There are four evaluation scripts in */scr*: eval.py comes from the original C-SWM source code; 
eval_ids.py checks for duplicate states, which cause the evaluation score to be lower than it should be;
  eval_ids_b.py partitions the evaluation set into batches and performs evaluation on each batch separately
  (which changes the score due to the number of states C-SWM ranks);
  eval_ids_b_inep.py evaluates the model's ability to distinguish states within a single episode (called "local" score in the paper).

# Citation

```
@article{biza21impact,
  title={The Impact of Negative Sampling on Contrastive Structured World Models}, 
  author={Ondrej Biza and Elise van der Pol and Thomas Kipf}, 
  journal={ICML 2021 Workshop: Self-Supervised Learning for Reasoning and Perception}, 
  year={2021} 
}
```