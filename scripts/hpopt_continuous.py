import argparse
import numpy as np
import optuna
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.experiments.single_env_experiment import SingleEnvExperiment
from all.presets import continuous



def evaluate_hp(hyperparameters):
    env = GymEnvironment('Pendulum-v0', device='cuda')
    agent = getattr(continuous, 'rrpg')
    agent = agent.device('cuda')
    agent = agent.hyperparameters(**hyperparameters)
    preset = agent.env(env).build()
    experiment = SingleEnvExperiment(
        preset,
        env,
        train_steps=100000,
        logdir='runs',
        quiet=True,
        render=False,
        write_loss=False,
        writer='tensorboard',
    )
    experiment.train(frames=200000)
    returns = experiment.test(episodes=100)
    experiment.close()

    return np.mean(returns)

def objective(trial):
    hyperparameters = {
        'lr_v': trial.suggest_loguniform('lr_v', 2e-6, 2e-1),
        'lr_pi': trial.suggest_loguniform('lr_pi', 1e-4, 1e-1),
    }
    return np.mean([evaluate_hp(hyperparameters) for i in range(5)])

def main():
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=40)
    print(study.best_params)


if __name__ == "__main__":
    main()
