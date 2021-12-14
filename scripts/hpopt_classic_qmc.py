import argparse
import numpy as np
import optuna
from all.experiments import run_experiment
from all.experiments.single_env_experiment import SingleEnvExperiment
from all.presets import classic_control
from all.qmc import QMCEngine, QMCCartPole_v1


def evaluate_hp(hyperparameters):
    # instanciate QMC engine
    # CartPole-v1's _max_episode_steps is 500
    qmc = QMCEngine(
        T_max=500,
        dim_action=1,
        dim_reseed=4,
        qmc_type='Sobol',
        pca_matrix=hyperparameters['pca_matrix'],
        scramble=True,
    )
    env = QMCCartPole_v1(qmc_engine=qmc, device='cpu')

    agent = classic_control.qmcpg
    agent = agent.device('cpu')
    agent = agent.hyperparameters(
        lr_v=hyperparameters['lr_v'],
        lr_pi=hyperparameters['lr_pi'],
        min_batch_size=hyperparameters['min_batch_size'],
        batch_reseeding=True
    )
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
    experiment.train(frames=50000)
    returns = experiment.test(episodes=100)
    experiment.close()

    return np.mean(returns)

def objective(trial):
    hyperparameters = {
        'lr_v': trial.suggest_loguniform('lr_v', 1e-6, 1e-1),
        'lr_pi': trial.suggest_loguniform('lr_pi', 1e-6, 1e-1),
        'min_batch_size': 2 ** trial.suggest_int('log2(min_batch_size)', 2, 6),
        'pca_matrix': trial.suggest_categorical('pca_matrix', ['ar0', 'ar1']),
    }
    return np.mean([evaluate_hp(hyperparameters) for i in range(5)])

def main():
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=40)
    print(study.best_params)


if __name__ == "__main__":
    main()
