import itertools
import os
from typing import Any

import pickle
import pandas as pd
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.dates as mdates



def plot_comparison(iters, methods: list, costs: list, average_costs: list,
                    stationary_classifier: pd.DataFrame, window: int, weights=None,
                    average_cost_bounds=None, title_prefix=""):
    """
    Plots the simulation results including costs, average costs, entropy, and average cost bounds.
    Parameters:
    - iters: List or array of iteration indices.
    - methods: the method names for the legend labels
    - costs: List of arrays of costs at each iteration.
    - average_costs: List of arrays of average costs up to each iteration.
    - average_cost_bounds: (Optional) List of arrays of average cost bounds up to each iteration.
    - title_prefix: (Optional) Prefix string for plot titles.
    - ax (Optional): the axis on which to plot
    """

    if average_cost_bounds is None:
        average_cost_bounds = [None for _ in average_costs]

    # fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    label_fs = 16
    tick_fs = 14
    legend_fs = 14
    fig = plt.figure(figsize=(22, 7))
    gs = GridSpec(2, 2)
    ax0 = plt.subplot(gs[:, 0])
    ax1 = plt.subplot(gs[1, 1])
    ax2 = plt.subplot(gs[0, 1])

    # grey colors for thresholds, cool for Boltzmann learning, warm for alternatives
    threshold_colors = ['black', 'grey', 'brown']
    boltzmann_colors = ['darkgreen', 'deepskyblue', 'green', 'lightblue', 'pink', 'purple']
    bench_colors = {'SVM': 'orange', 'MLP': 'red'}
    ssr_colors = ['lightblue', 'pink', 'darkgreen']

    # plot the stationary classifier performance
    if stationary_classifier is not None:
        timestamps = pd.to_datetime(stationary_classifier['timestamp'], unit='s')
        stationary_classifier.drop(columns=['timestamp'], inplace=True)
        rolled = stationary_classifier.rolling(window=window).mean()
        ssr_count = 0
        for i, column in enumerate(stationary_classifier.columns):
            if 'mae' in column:
                rolled[column].plot(label='Classifier {}'.format(column.split('_')[1]),
                                    color=ssr_colors[ssr_count], ax=ax1)
                ssr_count += 1
        cost_count = 0
        for idx, column in enumerate(stationary_classifier.columns):
            if 'cost' in column and 'c1' not in column:
                print('Column', stationary_classifier[column].mean())
                threshold = np.round(float(column.split('_')[-1]), 2)
                ax0.plot(iters, rolled[column], label='Threshold policy: {}'.format(threshold),
                                    linestyle='-.', color=threshold_colors[cost_count])
                cost_count += 1

    # Plot costs, average cost, and entropy, from start
    boltz_count = 0
    bench_count = 0
    for label, cost, average_cost, average_cost_bound in zip(methods, costs, average_costs, average_cost_bounds):
        print(label, np.mean(cost))
        rolling_cost = pd.Series(cost).rolling(window=window).mean()
        if 'Boltzmann' in label:
            color = boltzmann_colors[boltz_count]
            boltz_count += 1
        else:
            color = bench_colors[label]
            bench_count += 1
        ax0.plot(iters, rolling_cost, '-', label='{}'.format(label), color=color)
    freq = 25_000
    new_xticks = timestamps[0::freq].dt.to_period('D')
    ax0.set_xticks(new_xticks.index, new_xticks, rotation=45, ha='right')
    # Add secondary y-axis for weights
    # if weights is not None:
    #     for column in weights:
    #         if 'weight' in column:
    #             label = column.split('t')[1]
    #             ax2.plot(weights[column].rolling(window).mean(), label=r'P(a=1, c='+label+' | $\mathcal{F}$)', linewidth=1)

    xticks = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K')
    yticks_ax1 = ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
    ax1.yaxis.set_major_formatter(yticks_ax1)
    ax0.grid()
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(xticks)
        ax.tick_params(axis='both', labelsize=tick_fs)
        ax.grid()
    ax0.set_xlabel('Date', fontsize=label_fs)
    ax1.set_xlabel('Iteration', fontsize=label_fs)
    ax0.set_ylabel('Cost', fontsize=label_fs)
    ax1.set_ylabel('(Rolling) mean absolute error \n $|p_i - l_i|$', fontsize=label_fs)
    ax2.set_ylabel('Conditional action \n selection probability', fontsize=label_fs)  # Label the secondary y-axis for weights
    # ax0.set_ylim((0.03, .18))
    # ax1.set_ylim((0., .15))
    ax0.set_title(f"{title_prefix}")
    start_idx = timestamps[timestamps >= dt.datetime(2018, 1, 1)].index[0]
    end_idx = timestamps[timestamps <= dt.datetime(2019, 1, 1)].index[-1]
    ax0.set_xlim([start_idx, end_idx])

    # Add legends
    handles, labels = ax0.get_legend_handles_labels()
    try:
        def custom_key(item):
            if item == 'MLP':
                return 0
            elif item == 'SVM':
                return 1
            elif 'Threshold policy' in item:
                return 2
            elif 'Boltzmann Learning' in item:
                return 3
            return 4  # Default for any unhandled cases

        # Sort using the custom key
        sorted_pairs = sorted(zip(labels, handles), key=lambda x: custom_key(x[0]))
        sorted_labels, sorted_handles = zip(*sorted_pairs)
        ax0.legend(sorted_handles, sorted_labels, fontsize=legend_fs, loc='upper right')
    except:
        ax0.legend(fontsize=legend_fs, loc='upper right')
    ax1.legend(loc='center right', fontsize=legend_fs, bbox_to_anchor=(1.3, 0.5))
    ax2.legend(loc='center right', fontsize=legend_fs, bbox_to_anchor=(1.35, 0.5))  # Secondary legend for the weights

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(outfile[:-4] + '.png')
    plt.show()


def data_preprocessing(outcomes: dict[str, dict[str, Any]]):
    """
        Processes outcome data to prepare for simulation plotting.

        Args:
            outcomes (dict[str, dict[str, Any]]): dict containing simulation results.
                Each key is a method name, and the value is another dictionary with the following keys:
                - 'costs': np.array of cost values.
                - 'expected_costs': np.array of expected cost values.
                - 'iters': np.array of iteration indices.
                - 'entropy': np.array of entropy values.
                - 'energies': Dict[str, np.array] of energy distributions (optional for Boltzmann methods).

        Returns: (tuple)
                - iters (np.array): Array of iteration indices from the last processed method.
                - df (pd.DataFrame): DataFrame containing normalized energy weights for Boltzmann methods (or None if not applicable).
                - costs (list[np.array]): list of cost arrays for the selected methods.
                - average_costs (list[np.array]): list of expected cost arrays for the selected methods.
                - methods (list[str]): List of method names.
    """
    methods = []
    costs = []
    average_costs = []
    entropies = []
    min_bl_costs = [np.mean(entry['costs']) for key, entry in outcomes.items() if 'Boltzmann' in key]
    min_bl_costs.sort()
    df = None
    for key, entry in outcomes.items():
        print(key)

        if key == 'SVR':
            key = 'SVM'
        elif key == 'MLPRegressor':
            key = 'MLP'

        if np.any(np.mean(entry['costs']) <= min_bl_costs[0]) or 'Boltzmann' not in key:
            iters = entry['iters']
            methods.append(key)
            costs.append(entry['costs'])
            average_costs.append(entry['expected_costs'])
            entropies.append(entry['entropy'])

            if 'Boltzmann' in key and len(entry['energies']) > 0:
                beta = float(key.split('=')[-1].split('$')[0])
                df = pd.DataFrame.from_dict(entry['energies'], orient='columns')
                opts = [str(i) + str(j) + str(k) for i, j, k in itertools.product([0, 1], [0, 1], [0, 1])]
                df = np.exp(-beta*df)
                df = df.div(df.sum(axis=1), axis=0)
                # for opt in opts:
                #     df['weight' + opt] = df[opt + '1']

    return iters, df, costs, average_costs, methods


if __name__ == '__main__':

    type1_weight = 10.0
    type2_weight = 1.0
    n_train = 50000
    n_test = 350000
    header_timestamp: bool = True
    window_size = 100_000
    unit_step: bool = True
    if unit_step:
        step = '_unit'
    else:
        step = ''
    base_file = f'../data/ember_short{step}_{n_train}'
    classifier_file = f'../data/ember_short_unit_{n_train}{type1_weight}_{type2_weight}_short_classifier.pkl'
    file_dir = base_file + f'_{type1_weight}_{type2_weight}_ya2j_nonraw_noenergy'
    outfile = file_dir
    print(file_dir)

    sim_outcomes = dict()
    for file in os.listdir(file_dir):
        if not file == '.DS_Store':
            try:
                with open(file_dir + '/' + file, 'rb') as f:
                    sim_outcomes[file.split('.pkl')[0]] = pickle.load(f)
            except EOFError:
                print(f'Unable to open {file}')

    # load data
    with open(classifier_file, 'rb') as f:
        classifier_perf = pickle.load(f)

    # with open(filename, 'rb') as f:
    #     try:
    #         sim_outcomes = pickle.load(f)
    #     except:
    #         sim_outcomes, _ = pickle.load(f)

    x_iters, processed_data, costs, average_costs, methods_list = data_preprocessing(outcomes=sim_outcomes)

    plot_comparison(iters=x_iters, methods=methods_list,
                    costs=costs,
                    average_costs=average_costs,
                    average_cost_bounds=None,
                    title_prefix='',
                    weights=processed_data,
                    stationary_classifier=classifier_perf,
                    window=window_size)
