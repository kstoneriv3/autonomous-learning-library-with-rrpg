import argparse
from all.experiments import run_experiment
from all.presets import classic_control
from all.qmc import QMCEngine, QMCCartPole_v1
from utils import str2bool


def main():
    parser = argparse.ArgumentParser(description="Run a classic control benchmark.")
    parser.add_argument("--qmc_type", type=str, default='Sobol', help="Name of the QMC sequence.")
    parser.add_argument("--pca_matrix", type=str, default='ar1', help="ar0, ar1, ar2.")
    parser.add_argument("--scramble", type=str2bool, default=True)
    parser.add_argument("--batch_reseeding", type=str2bool, default=True)
    parser.add_argument("--quiet", type=str2bool, default=False)
    # parser.add_argument("--min_batch_size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=50000, help="The number of training frames."
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment."
    )
    parser.add_argument(
        "--logdir", default='runs', help="The base logging directory."
    )
    parser.add_argument("--writer", default='tensorboard', help="The backend used for tracking experiment metrics.")
    parser.add_argument(
        '--hyperparameters',
        default=[],
        nargs='*',
        help="Custom hyperparameters, in the format hyperparameter1=value1 hyperparameter2=value2 etc."
    )
    args = parser.parse_args()

    # instanciate QMC engine
    # CartPole-v1's _max_episode_steps is 500
    qmc = QMCEngine(
        T_max=500,
        dim_action=1,
        dim_reseed=4,
        qmc_type=args.qmc_type,
        pca_matrix=args.pca_matrix,
        scramble=args.scramble,
    )
    env = QMCCartPole_v1(qmc_engine=qmc, device=args.device)

    agent = classic_control.qmcpg
    agent = agent.device(args.device)


    # parse hyperparameters
    hyperparameters = {
        'batch_reseeding': args.batch_reseeding,
        # 'min_batch_size': args.min_batch_size,
    }
    for hp in args.hyperparameters:
        key, value = hp.split('=')
        hyperparameters[key] = type(agent.default_hyperparameters[key])(value)
    agent = agent.hyperparameters(**hyperparameters)

    run_experiment(
        agent,
        env,
        frames=args.frames,
        render=args.render,
        logdir=args.logdir,
        writer=args.writer,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
