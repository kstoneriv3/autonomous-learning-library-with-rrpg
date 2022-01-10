import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

methods = ["vpg", "rrpg", "qmcpg"]


def main():
    parser = argparse.ArgumentParser(description="Run a classic control benchmark.")
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--max_frame", type=int, default=200000)
    parser.add_argument("--n_mesh", type=int, default=200)
    args = parser.parse_args()
    log_dir = args.log_dir
    out_dir = args.out_dir
    max_frame = args.max_frame
    n_mesh = args.n_mesh

    buckets = np.linspace(0, max_frame, n_mesh + 1)

    returns = {m:[] for m in methods}
    discounted_returns = {m:[] for m in methods}
    grad_var = {m:[] for m in methods}
    grad_norm2 = {m:[] for m in methods}
    grad_cost = {m:[] for m in methods}
    grad_var_times_cost = {m:[] for m in methods}
    grad_expection_norm2 = {m:[] for m in methods}

    for d in os.listdir(log_dir):
        # find the method for the path (log) by the first 3 chars of the path
        method = {
            "vpg": "vpg",
            "rrp": "rrpg",
            "qmc": "qmcpg",
        }[d[:3]]

        # extend `d` to a path
        d = os.path.join(log_dir, d, "CartPole-v1")

        # read csv files as numpy arrays
        returns[method].append(bucket_average(os.path.join(d, "returns.csv"), buckets))
        discounted_returns[method].append(bucket_average(os.path.join(d, "discounted_returns.csv"), buckets))
        grad_norm2[method].append(bucket_average(os.path.join(d, "grad_norm.csv"), buckets))
        # this makes it hard to use for loop different types of plots
        norm2 = bucket_average(os.path.join(d, "grad_norm.csv"), buckets)
        var = bucket_average(os.path.join(d, "grad_var.csv"), buckets)
        cost = bucket_average(os.path.join(d, "grad_cost.csv"), buckets)
        grad_var[method].append(var)
        grad_cost[method].append(cost)
        grad_norm2[method].append(norm2)
        grad_var_times_cost[method].append([v * c for v, c in zip(var, cost)])
        grad_expection_norm2[method].append([np.sqrt(n - v) ** 2 for n, v in zip(norm2, var)])

    plot_with_confint(buckets, returns, "(Undiscounted) Cumulative Reward", out_dir) 
    plot_with_confint(buckets, discounted_returns, "Discounted Cumulative Reward", out_dir) 
    plot_with_confint(buckets, grad_norm2, "Squared Norm of Batch Gradient", out_dir, log=True) 
    plot_with_confint(buckets, grad_var, "Variance of Batch Gradient", out_dir, log=True) 
    plot_with_confint(buckets, grad_cost, "Cost of a Batch", out_dir, log=True) 
    plot_with_confint(buckets, grad_var_times_cost, "Variance for Unit Cost", out_dir, log=True) 
    plot_with_confint(buckets, grad_expection_norm2, "Squared Norm of Expected Gradient", out_dir, log=True) 

def bucket_average(csv_path, buckets):
    data = pd.read_csv(csv_path, header=None)
    frames = data.iloc[:, 0].to_numpy()
    values = data.iloc[:, 1].to_numpy()
    if csv_path[-8:-4] == "norm":
        values = values ** 2
    sep_indices = np.searchsorted(frames, buckets)
    means = []
    for start, end in zip(sep_indices[:-1], sep_indices[1:]):
        means.append(values[start:end].mean() if start < end else np.nan)
    return means

def plot_with_confint(buckets, data, y_label, out_dir, log=False):
    bucket_centers = 0.5 * buckets[1] + buckets[:-1]

    plt.style.use("seaborn-whitegrid")
    plt.set_cmap("tab10")
    legend_args  = [[], []]
    for method in methods:
        values = np.array(data[method])
        if log:
            log_values = np.log(values)
            means = np.nanmean(log_values, axis=0)
            std = np.nanstd(log_values, axis=0)  / np.sqrt(np.sum(~np.isnan(values), axis=0)) * 2.5
            lower, upper = means - std, means + std
            means, lower, upper = map(np.exp, [means, lower, upper])
        else:
            means = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)  / np.sqrt(np.sum(~np.isnan(values), axis=0)) * 2.5
            lower, upper = means - std, means + std
        line = plt.plot(bucket_centers, means)[0]
        band = plt.fill_between(bucket_centers, lower, upper, alpha=0.1)
        legend_args[0].append((line, band))
        legend_args[1].append(method.upper())

    plt.legend(*legend_args)  # ignore fill_between
    plt.xlabel("Total frames")
    plt.ylabel(y_label)
    plt.xlim([0, buckets[-1]])
    if log:
        plt.yscale("log")
    plt.savefig(os.path.join(out_dir, y_label + ".pdf"))
    plt.clf()

if __name__ == "__main__":
    main()
