# The Autonomous Learning Library: A PyTorch Library for Building Reinforcement Learning Agents

This repository contains the implementation of Russian Roulette policy gradient (RRPG) estimator as an extension to the `autonomous-learning-library`, which is a deep reinforcement learning (DRL) library for PyTorch.

## Installation

You can install the `autonomous-learning-library` with RRPG directly from this repository:

```
git clone https://github.com/kstoneriv3/autonomous-learning-library-with-rrpg.git
cd autonomous-learning-library-with-rrpg
pip install -e .
```

## How to Run the Experiments

### Running the VPG, RRPG, and QMCPG on CartPole-v1
If you want to run vanilla policy gradient (VPG), Russian roulette policy gradient, or Quasi Monte Carlo policy gradient (QMCPG) for 1000000 frames, you can run the following scripts at the root directory.

```
python scripts/classic.py CartPole-v1 vpg --frames 1000000 --quiet True --device cpu
python scripts/classic.py CartPole-v1 rrpg --frames 1000000  --quiet True --device cpu
python scripts/classic_qmc.py --frames 1000000 --quiet True --device cpu
```

These scripts will run the experiments and outputs the logs (e.g. discounted cumulative reward, statistics on gradients) to `./runs`. 

### Plotting the Experimental Results
You can plot the results of the experiments at `./runs` and save them at `./out` by the running the following script.
```
python scripts/plot_learning_curve.py --log_dir ./runs --out_dir ./out --max_frame 200000
```

### Running Experiments in Parallel
When you want to run experiments in parallel, you can run multiple experiments at once. You can find example scripts for parallel experiments at `./scripts/run_all.sh` and `./scripts/run_all_euler.sh`.
