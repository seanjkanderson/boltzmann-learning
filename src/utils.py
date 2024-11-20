import itertools
import time
from copy import deepcopy, copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_simulation_results(iters, costs, average_costs, entropies, average_cost_bounds=None, title_prefix=""):
    """
    Plots the simulation results including costs, average costs, entropy, and average cost bounds.

    Parameters:
    - iters: List or array of iteration indices.
    - costs: List or array of costs at each iteration.
    - average_costs: List or array of average costs up to each iteration.
    - entropies: List of players' array of entropy values at each iteration.
    - average_cost_bounds: (Optional) List or array of average cost bounds up to each iteration.
    - title_prefix: (Optional) Prefix string for plot titles.
    """
    slc = slice(int(.95 * len(iters)), len(iters))

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Plot costs, average cost, and entropy, from start
    ax_twin = ax[0].twinx()
    lines = ax[0].plot(iters, costs, '.', label='cost', color='tab:blue')
    lines += ax[0].plot(iters, average_costs, '-', label='average cost', color='tab:orange')
    if average_cost_bounds is not None:
        lines += ax[0].plot(iters, average_cost_bounds, '-', label='average cost bound', color='tab:cyan')
    # for idx, entropy in enumerate(entropies):
    #     lines += ax_twin.plot(iters, entropy, '.', label='entropy: P{}'.format(idx+1), color='tab:green')

    ax[0].grid()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('cost')
    ax[0].set_ylim((-1.1, 1.1))
    ax_twin.set_ylabel('entropy')
    ax[0].set_title(f"{title_prefix} Costs, Average Costs, and Entropy (Start)")
    ax_twin.legend(lines, [line.get_label() for line in lines])

    # Plot costs, average cost, and entropy, skipping start
    ax_twin = ax[1].twinx()
    lines = ax[1].plot(iters[slc], average_costs[slc], '-', label='average cost', color='tab:orange')
    if average_cost_bounds is not None:
        lines += ax[1].plot(iters[slc], average_cost_bounds[slc], '-', label='average cost bound', color='tab:cyan')
    for idx, entropy in enumerate(entropies):
        lines += ax_twin.plot(iters[slc], entropy[slc], '.', label='entropy: P{}'.format(idx+1), color='tab:green')

    ax[1].grid()
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('cost')
    ax_twin.set_ylabel('entropy')
    ax[1].set_title(f"{title_prefix} Costs, Average Costs, and Entropy (End)")
    ax_twin.legend(lines, [line.get_label() for line in lines])

    ax_twin.set_zorder(ax[0].get_zorder() + 1)
    fig.tight_layout()
    plt.show()


def plot_comparison(iters, methods: list, costs: list, average_costs: list, entropies: list,
                    average_cost_bounds=None, title_prefix="", ax=None):
    """
    Plots the simulation results including costs, average costs, entropy, and average cost bounds.
    # TODO: clean up this doc
    Parameters:
    - iters: List or array of iteration indices.
    - methods: the method names for the legend labels
    - costs: List of arrays of costs at each iteration.
    - average_costs: List of arrays of average costs up to each iteration.
    - entropies: List of players' array of entropy values at each iteration.
    - average_cost_bounds: (Optional) List of arrays of average cost bounds up to each iteration.
    - title_prefix: (Optional) Prefix string for plot titles.
    - ax (Optional): the axis on which to plot
    """

    if average_cost_bounds is None:
        average_cost_bounds = [None for _ in average_costs]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plot costs, average cost, and entropy, from start
    for label, cost, average_cost, average_cost_bound in zip(methods, costs, average_costs, average_cost_bounds):
        print(label)
        # lines = ax.plot(iters, cost, '.', label='cost | {}'.format(label))
        lines = ax.plot(iters, average_cost, '-', label='average cost | {}'.format(label))
        if average_cost_bound is not None:
            lines += ax.plot(iters, average_cost_bound, '-', label='average cost bound | {}'.format(label))

        ax.grid()
        ax.set_xlabel('t')
        ax.set_ylabel('cost')
        ax.set_ylim((-1.1, 1.1))
        ax.set_title(f"{title_prefix}")

    ax.scatter(range(len(entropies[0])), entropies[0], label='LearningGame entropy', s=1)
    ax.legend()
    # fig.tight_layout()
    plt.show()


def generate_probabilities_matrix(prob_matrix: np.array) -> (list[str], np.array):
    """
    Given probabilities (row is a sample, and columns correspond to independent probabilities), compute the probability
    of each possible outcome. Assumes a binary setting, resulting in 2^{n_columns} possible outcomes.
    Args:
        prob_matrix (np.array): contains the

    Returns:

    """
    n = prob_matrix.shape[1]  # Number of probabilities in each set

    # Generate all possible outcomes (binary strings of length n)
    outcomes = np.array(list(itertools.product([0, 1], repeat=n)))

    # Convert the probabilities list to a NumPy array
    prob_matrix = np.array(prob_matrix)

    # Calculate the probability of each outcome for each set of probabilities
    outcome_probabilities = []
    for probs in prob_matrix:
        # Compute the probabilities for this set
        probs_matrix = probs * outcomes + (1 - probs) * (1 - outcomes)
        outcome_probs = np.prod(probs_matrix, axis=1)
        outcome_probabilities.append(outcome_probs)

    outcomes_strings = ["".join(outcome.astype(str)) for outcome in outcomes]
    return outcomes_strings, np.array(outcome_probabilities)


class GamePlay:
    """Creates a game for each decision maker to play. Runs in series, but could be parallelized."""
    def __init__(self, decision_makers: list, game, horizon: int,
                 disp_results_per_iter: int, store_energy_hist: bool, binary_cont_measurement: bool = False):
        """
        Args:
            decision_makers (list): decision-making algorithms to be tested
            game: an instance of a game to play # TODO: should specify type
            horizon (int): the horizon of the game
            disp_results_per_iter (int): how often to output current gameplay info (1 is every step)
            binary_cont_measurement (bool): whether to plot current policy at each disp_results_per_iter. Only supported
                                            for binary continuous measurement settings.
            store_energy_hist (bool): whether to keep track of the energy at each time step. Can take significant memory
        """
        self.decision_makers = decision_makers  # each decision maker will
        self.game = game
        self.horizon = horizon
        self.disp_results_per_iter = disp_results_per_iter
        self.binary_cont_measurement = binary_cont_measurement
        self._store_energy_hist = store_energy_hist

    def play_games(self) -> dict[dict]:
        """ Play all the games
        Returns: dictionary of all game outcomes, which are dictionaries of relevant statistics

        """
        outputs = dict()
        for decision_maker in self.decision_makers:
            game_i = deepcopy(self.game)  # create a copy of the game
            algorithm_name = decision_maker.__class__.__name__
            if algorithm_name == 'SklearnModel':
                algorithm_name = decision_maker.model.__class__.__name__
            print('Playing', algorithm_name)
            if algorithm_name == 'LearningGame':
                algorithm_name = r'Boltzmann Learning | $\lambda={:.1e}, \beta={:.1e}$'\
                    .format(decision_maker.decay_rate, decision_maker.inverse_temperature)
            outputs[algorithm_name] = self._play_game(game_i, decision_maker)
        print('Finished playing games.')
        return outputs

    def _play_game(self, game, decision_maker) -> dict:
        """
        Play a particular game. Helper function for self.play_games()
        Args:
            game: the game to play  #TODO: needs typing
            decision_maker (decision_maker.DecisionMaker): the agent to make decisions #TODO: needs typing

        Returns: (dict) containing a summary of relevant statistics from the simulation

        """

        costs = np.zeros(self.horizon)
        expected_costs = np.zeros(self.horizon)
        cost_bounds = np.zeros(self.horizon)
        entropy = np.zeros(self.horizon)
        p1_action = []
        probs = []
        energies = dict()
        start_time = time.perf_counter()
        for idx in range(self.horizon - 1):
            # Play
            measurement, raw_measurement = game.get_measurement()
            action, prob, entropy[idx] = decision_maker.get_action(measurement=measurement, time=idx,
                                                                   raw_measurement=raw_measurement)
            # print('action time {}'.format(time.perf_counter() - st))
            costs[idx], all_costs, opponent_action = game.play(action)
            p1_action.append(action)
            probs.append(prob)
            if self._store_energy_hist:
                if hasattr(decision_maker, 'energy'):
                    # TODO: should be a cleaner way to do this but gets around immutability of dict
                    for key, val in decision_maker.energy.items():
                        for key2, val2 in val.items():
                            total_key = key + str(key2)
                            if total_key not in energies.keys():
                                energies[total_key] = []
                            energies[total_key].append(val2)
            if prob is None:
                expected_costs[idx] = None
            else:
                expected_costs[idx] = (np.array([v for k, v in all_costs.items()]) * prob).sum()
            # Learn
            decision_maker.update_energies(measurement=measurement, costs=all_costs, action=action,
                                           raw_measurement=raw_measurement,
                                           time=idx, action_cost=costs[idx], opponent_action=opponent_action)
            # Store regret
            try:
                # TODO: ideally all decision makers will have regret function
                (_, _, cost_bounds[idx], _, _, _, _, _) = decision_maker.get_regret(display=False)
            except Exception:
                pass

            # Output
            if (idx % self.disp_results_per_iter == 0) or (idx == self.horizon - 1):
                print(idx)
                if idx > 0:
                    elapsed_time = time.perf_counter() - start_time
                    print('Elapsed time: {} | Average per step time: {}'.format(elapsed_time, elapsed_time / idx))

                if self.binary_cont_measurement:
                    plot_binary_policy(decision_maker=decision_maker, game=game, time_idx=idx)
        elapsed_time = time.perf_counter() - start_time
        print('Total time: {} | Average per step time: {}'.format(elapsed_time, elapsed_time/self.horizon))
        iters = range(self.horizon)
        moving_average_costs = pd.Series(costs).rolling(window=1000, min_periods=1).mean()
        return dict(iters=iters, costs=costs, cost_bounds=cost_bounds, expected_costs=expected_costs,
                    average_costs=moving_average_costs, entropy=entropy, probs=probs, energies=energies)


def plot_binary_policy(decision_maker, game, time_idx):
    """Plot the action selection probabilities for a binary setting (i.e. |A|=2)."""
    fig, ax = plt.subplots()
    p = []
    meas_range = np.arange(0.1, 1.0, step=.01).round(2)
    pos_res = []
    bin_counts = []
    meas_seq_probs = game.measurement_sequence[:, 1].round(2)
    for measurement_m in meas_range:
        (_, prob_m, _) = decision_maker.get_action(measurement={'0': 1 - measurement_m, '1': measurement_m}, time=time_idx)
        p.append(prob_m[1])
        instance_count = (meas_seq_probs == measurement_m).sum()
        bin_counts.append(instance_count)
        pos_res.append(np.sum(game.opponent_action_sequence[meas_seq_probs == measurement_m] == 1) / instance_count)
    p = np.array(p)
    pos_res = np.array(pos_res)

    ax[0].plot(meas_range, p, label='Boltzmann')
    ax[0].plot(meas_range, pos_res, label='percent malicious')
    ax[0].legend()
    ax[0].set_ylabel('Prob action="mark malicious"')
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlim(0, 1.)
    ax[1].set_xlabel('Measurement (prob malicious)')
    ax[0].grid()
    ax[1].scatter(meas_range, bin_counts)
    ax[1].set_ylabel('Bin counts')
    plt.show()