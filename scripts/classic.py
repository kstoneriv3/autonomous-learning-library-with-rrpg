import argparse
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.presets import classic_control
from utils import str2bool

def main():
    parser = argparse.ArgumentParser(description="Run a classic control benchmark.")
    parser.add_argument("env", help="Name of the env (e.g. CartPole-v1).")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument("--quiet", type=str2bool, default=False)
    #parser.add_argument("--min_batch_size", type=int, default=16)
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

    env = GymEnvironment(args.env, device=args.device)

    agent_name = args.agent
    agent = getattr(classic_control, agent_name)
    agent = agent.device(args.device)

    # parse hyperparameters
    hyperparameters = {} #{'min_batch_size': args.min_batch_size}
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
