import argparse
#from joblib import Parallel, delayed
import numpy as np
import optuna
from all.experiments import run_experiment
from all.experiments.single_env_experiment import SingleEnvExperiment
from all.presets import classic_control
from all.qmc import QMCEngine, QMCCartPole_v1

DEVICE = 'cuda'
#DEVICE = 'cpu'

def evaluate_hp(hyperparameters):
    # instanciate QMC engine
    # CartPole-v1's _max_episode_steps is 500
    qmc = QMCEngine(
        T_max=500,
        dim_action=1,
        dim_reseed=4,
        qmc_type='Random', #'Sobol',
        pca_matrix=hyperparameters['pca_matrix'],
        scramble=True,
    )
    env = QMCCartPole_v1(qmc_engine=qmc, device=DEVICE)

    agent = classic_control.qmcpg
    agent = agent.device(DEVICE)
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
        train_steps=200000,
        logdir='runs',
        quiet=True,
        render=False,
        write_loss=False,
        writer='tensorboard',
    )
    #experiment.train(frames=100000)
    experiment.train(frames=200000)
    returns = experiment.test(episodes=100)
    experiment.close()

    return np.mean(returns)

def objective(trial):
    '''
    hyperparameters = {
        'lr_v': trial.suggest_loguniform('lr_v', 1e-6, 1e-1),
        'lr_pi': trial.suggest_loguniform('lr_pi', 1e-6, 1e-1),
        'min_batch_size': 2 ** trial.suggest_int('log2(min_batch_size)', 2, 6),
        'pca_matrix': trial.suggest_categorical('pca_matrix', ['ar0', 'ar1']),
    }
    # train_steps = 200000
    # best is trial 31/40 with value: 338.944. 
    # {'lr_v': 1.0605694257370513e-05, 'lr_pi': 0.011508029664067079, 'log2(min_batch_size)': 2, 'pca_matrix': 'ar0'
    # Reduces the batch size to increase the number of updates.
    '''

    hyperparameters = {
        'lr_v': trial.suggest_loguniform('lr_v', 1e-6, 1e-3),
        'lr_pi': trial.suggest_loguniform('lr_pi', 5e-3, 5e-2),
        'min_batch_size': 16,
        'pca_matrix': 'ar1',
    }
    # For Random case, 
    # Trial 11 finished with value: 391.56 and parameters: 
    # {'lr_v': 1.0534936092129222e-06, 'lr_pi': 0.02050275584817121}.
    # 
    # For Sobol case,
    # Best is trial 4 with value: 473.524.
    # {'lr_v': 1.3016673640012752e-05, 'lr_pi': 0.027928126384815857}

    return np.mean([evaluate_hp(hyperparameters) for i in range(5)])


def main():
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=40)
    print(study.best_params)


if __name__ == "__main__":
    main()
