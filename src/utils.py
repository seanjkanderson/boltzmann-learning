import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


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


def plot_comparison(iters, methods: list, costs: list, average_costs: list, entropies: list, average_cost_bounds=None, title_prefix=""):
    """
    Plots the simulation results including costs, average costs, entropy, and average cost bounds.

    Parameters:
    - iters: List or array of iteration indices.
    - methods: the method names for the legend labels
    - costs: List of arrays of costs at each iteration.
    - average_costs: List of arrays of average costs up to each iteration.
    - entropies: List of players' array of entropy values at each iteration.
    - average_cost_bounds: (Optional) List of arrays of average cost bounds up to each iteration.
    - title_prefix: (Optional) Prefix string for plot titles.
    """

    if average_cost_bounds is None:
        average_cost_bounds = [None, None, None]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plot costs, average cost, and entropy, from start
    for label, cost, average_cost, average_cost_bound in zip(methods, costs, average_costs, average_cost_bounds):
        lines = ax.plot(iters, cost, '.', label='cost | {}'.format(label))
        lines += ax.plot(iters, average_cost, '-', label='average cost | {}'.format(label))
        if average_cost_bound is not None:
            lines += ax.plot(iters, average_cost_bound, '-', label='average cost bound | {}'.format(label))
        # for idx, entropy in enumerate(entropies):
        #     lines += ax_twin.plot(iters, entropy, '.', label='entropy: P{}'.format(idx+1), color='tab:green')

        ax.grid()
        ax.set_xlabel('t')
        ax.set_ylabel('cost')
        ax.set_ylim((-1.1, 1.1))
        ax.set_title(f"{title_prefix} Costs, Average Costs, and Entropy (Start)")

        # # Plot costs, average cost, and entropy, skipping start
        # ax_twin = ax[1].twinx()
        # lines = ax[1].plot(iters[slc], average_cost[slc], '-', label='average cost | {}'.format(label))
        # if average_cost_bound is not None:
        #     lines += ax[1].plot(iters[slc], average_cost_bound[slc], '-', label='average cost bound | {}'.format(label))
        # for idx, entropy in enumerate(entropies):
        #     lines += ax_twin.plot(iters[slc], entropy[slc], '.', label='entropy: P{} | {}'.format(idx+1, label))
        #
        # ax[1].grid()
        # ax[1].set_xlabel('t')
        # ax[1].set_ylabel('cost')
        # ax_twin.set_ylabel('entropy')
        # ax[1].set_title(f"{title_prefix} Costs, Average Costs, and Entropy (End)")
        # ax_twin.legend(lines, [line.get_label() for line in lines])

    ax.legend()
    fig.tight_layout()
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

    def __init__(self, decision_makers: list, game, horizon: int,
                 disp_results_per_iter: int, binary_cont_measurement: bool = False):
        """ creates a parallel game for each decision maker
        Args:
            decision_makers (list): decision-making algorithms to be tested # TODO should create an abstract class for this?
            game: an instance of a game to play # TODO: should specify type
            horizon (int): the horizon of the game
            disp_results_per_iter (int): how often to output current gameplay info (1 is every step)
            binary_cont_measurement (bool): whether to plot current policy at each disp_results_per_iter. Only supported
                                            for binary continuous measurement settings.
        """
        self.decision_makers = decision_makers  # each decision maker will
        self.games = [deepcopy(game) for _ in decision_makers]  # create an identical game for each decision maker
        self.horizon = horizon
        self.disp_results_per_iter = disp_results_per_iter
        self.binary_cont_measurement = binary_cont_measurement

    def play_games(self) -> dict[dict]:
        """ Play all of the games
        Returns: dictionary of all game outcomes, which are dictionaries of relevant statistics

        """
        outputs = dict()
        for game, decision_maker in zip(self.games, self.decision_makers):
            algorithm_name = decision_maker.__class__.__name__
            print('Playing', algorithm_name)
            outputs[algorithm_name] = self._play_game(game, decision_maker)
        return outputs

    def _play_game(self, game, decision_maker) -> dict:

        costs = np.zeros(self.horizon)
        cost_bounds = np.zeros(self.horizon)
        entropy = np.zeros(self.horizon)
        p1_action = []

        for idx in range(self.horizon - 1):
            # Play
            measurement = game.get_measurement()
            action, prob, entropy[idx] = decision_maker.get_action(measurement, idx)
            costs[idx], all_costs, opponent_action = game.play(action)
            p1_action.append(action)
            # Learn
            decision_maker.update_energies(measurement=measurement, costs=all_costs, action=action,
                                           time=idx, action_cost=costs[idx], opponent_action=opponent_action)
            # Store regret
            try:
                # TODO: ideally all decision makers with have regret function
                (_, _, cost_bounds[idx], _, _, _, _, _) = decision_maker.get_regret(display=False)
            except Exception:
                pass

            # Output
            if (idx % self.disp_results_per_iter == 0) or (idx == self.horizon - 1):
                # print("iter={:4d}, action_1 = {:2.0f}, " # TODO: generalize this printout for different action types
                #       "cost = {:2.0f}, "
                #       "all_costs = {:s}".format(idx,
                #                                 action,
                #                                 # p2_act_sequence[
                #                                 #     idx],
                #                                 costs[
                #                                     idx],
                #                                 str(all_costs)))
                # print('measurement: {}'.format(str(measurement)))
                # print("energy: {}".format(decision_maker.energy))

                if self.binary_cont_measurement:
                    plot_binary_policy(decision_maker=decision_maker, game=game, time_idx=idx)

        iters = range(self.horizon)
        average_costs = np.divide(np.cumsum(costs), range(1, self.horizon + 1))
        average_cost_bounds = np.divide(np.cumsum(cost_bounds), np.add(range(self.horizon), 1))
        print('Average costs: {} | Average cost bounds: {}'.format(average_costs[-10:].mean(),
                                                                   average_cost_bounds.mean()))
        return dict(iters=iters, costs=costs, average_costs=average_costs, entropy=entropy)

    @staticmethod
    def plot_game_outcome(outcomes):

        methods = []
        costs = []
        average_costs = []
        entropies = []
        for key, entry in outcomes.items():
            iters = entry['iters']
            methods.append(key)
            costs.append(entry['costs'])
            average_costs.append(entry['average_costs'])
            entropies.append(entry['entropy'])
        plot_comparison(iters=iters, methods=methods,
                        costs=costs,
                        average_costs=average_costs,
                        entropies=entropies,
                        average_cost_bounds=None,
                        title_prefix="MAB | Bad RNG")


def plot_binary_policy(decision_maker, game, time_idx):

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
    # ax[0].axvline(x=classifier_threshold, c='red', label='classifier threshold')
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