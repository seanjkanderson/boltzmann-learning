import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_simulations(df: pd.DataFrame,
                     iters: np.ndarray,
                     window: int,
                     boltzmann_colors: list[str],
                     benchmark_colors: list[str],
                     iterations_up_to: int,
                     label_fs: int,
                     tick_fs: int,
                     legend_fs: int,
                     file_to_write: str,
                     linewidth: float = 1
                     ) -> None:
    """
    Plots simulation results as rolling costs for different decision-making methods

    Args:
        df (pd.DataFrame): DataFrame containing columns ['methods', 'costs', 'average_costs', 'entropies']
        iters (np.array): Array of iteration indices
        window (int): Rolling window size for smoothing costs
        boltzmann_colors (list[str]): List of colors for Boltzmann-based methods
        benchmark_colors (list[str]): List of colors for benchmark methods
        iterations_up_to (int): Number of iterations to display on the plot
        label_fs (int): Font size for axis labels
        tick_fs (int): Font size for axis ticks
        legend_fs (int): Font size for the legend
        file_to_write (str): File path to save the resulting plot
        linewidth (float): Line width for plot lines
            Defaults to 1
    """
    fig, ax0 = plt.subplots(figsize=(12, 7))
    # Plot costs, average cost, and entropy, from start
    name_map = {'PrescientBayes': 'Prescient Bayesian Estimator',
                'BayesianEstimator': 'Bayesian Estimator',
                'SVR': 'SVM', 'SVC': 'SVM', 'MLPRegressor': 'MLP', 'MLPClassifier': 'MLP'}
    boltz_count = 0
    bench_count = 0
    for idx, (label, cost, average_cost, entropy) in \
            enumerate(zip(df['methods'], df['costs'], df['average_costs'], df['entropies'])):

        if label in name_map.keys():
            label = name_map[label]
        print(label, np.mean(cost))
        rolling_cost = pd.Series(cost).rolling(window=window).mean()

        if 'Boltzmann' in label:
            color = boltzmann_colors[boltz_count]
            boltz_count += 1
        else:
            color = benchmark_colors[bench_count]
            bench_count += 1
        ax0.plot(iters[:iterations_up_to], rolling_cost.iloc[:iterations_up_to], '-',
                 label=label, color=color, linewidth=linewidth)

    xticks = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K')
    ax0.xaxis.set_major_formatter(xticks)
    ax0.grid()
    ax0.set_xlabel('Iteration', fontsize=label_fs)
    ax0.set_ylabel('Cost', fontsize=label_fs)
    ax0.set_ylim((-.88, 0))
    ax0.tick_params(axis='both', labelsize=tick_fs)

    handles, labels = ax0.get_legend_handles_labels()
    try:
        order = [0, 1, 3, 2, 4, 5, 6]
        ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=legend_fs,
                   loc='upper right')
    except IndexError:
        ax0.legend()

    fig.tight_layout()

    fig.savefig(file_to_write + '.png')
    plt.show()


def data_preprocessing(outcomes: dict[str, dict[str, np.ndarray]]
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess simulation outcomes to extract methods, costs, average costs, lambda values.

    Args:
        outcomes (dict[str, dict[str, np.ndarray]]): keys are method names and values are
            dicts containing simulation data with the following keys:
            - 'costs': Array of cost values for the method.
            - 'expected_costs': Array of expected cost values for the method.
            - 'iters': Array of iteration indices.

    Returns:
        tuple[np.ndarray, pd.DataFrame]:
            - `iters`: array of iteration indices, common for all methods.
            - `df`: Pandas DataFrame containing the preprocessed data with the following columns:
                - 'methods': decision maker names.
                - 'costs': List of cost arrays for each method.
                - 'average_costs': List of average cost arrays for each method.
                - 'entropies': List of entropy arrays for each method.
                - 'lambda_val': Lambda values extracted from method names.
    """
    data = dict(methods=[], costs=[], average_costs=[], entropies=[], lambda_val=[])
    min_bl_costs = [np.mean(entry['costs']) for key, entry in outcomes.items() if 'Boltzmann' in key]
    min_bl_costs.sort()
    for key, entry in outcomes.items():
        print(key)
        try:
            lam = key.split('=')[1].split(',')[0]
            bet = key.split('=')[-1]
        except IndexError:
            lam = 0.
            pass
        if np.any(np.mean(entry['costs']) <= min_bl_costs[:1]) or 'Boltzmann' not in key \
                or ('0.0' in lam and '1.0e+00' in bet) \
                or ('1.0e-02' in lam and '1.0e+00' in bet) \
                or ('1.0e-03' in lam and '1.0e+01' in bet):  # second condition is only needed for paper comparisons
            iters = entry['iters']
            data['methods'].append(key)
            data['costs'].append(entry['costs'])
            data['average_costs'].append(entry['expected_costs'])
            data['entropies'].append(entry['entropy'])
            data['lambda_val'].append(float(lam))

    df = pd.DataFrame.from_dict(data, orient='columns')
    df = df.sort_values('lambda_val')
    return iters, df


if __name__ == '__main__':
    # specify filename of the simulation results. Leave off file extension as this will also be used for naming the
    # resulting image generated (i.e. file = 'test' will read read 'test.pkl' and save an image 'test.png'
    file = '../data/rps_102000_False'
    with open(file + '.pkl', 'rb') as f:
        sim_outcomes, _ = pickle.load(f)

    rolling_window = 1_000  # rolling average window size
    label_fs = 16  # font size for figure labels
    tick_fs = 14  # font size for ticks
    legend_fs = 14  # legend font size
    iterations_up_to = 100000  # iterations to include in results. Can be useful for focusing plot on early part of sim.

    # colors for the lines for the results from boltzmann learning methods
    boltzmann_colors = ['blue', 'deepskyblue', 'green', 'limegreen']
    # colors for the lines from alternative/benchmark methods
    benchmark_colors = ['red', 'orange', 'purple']

    x_iters, formatted_data = data_preprocessing(sim_outcomes)
    plot_simulations(df=formatted_data,
                     iters=x_iters,
                     window=rolling_window,
                     boltzmann_colors=boltzmann_colors,
                     benchmark_colors=benchmark_colors,
                     iterations_up_to=iterations_up_to,
                     label_fs=label_fs,
                     tick_fs=tick_fs,
                     legend_fs=legend_fs,
                     file_to_write=file)
