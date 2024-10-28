import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class RPS_vs_bad_rng:
    action_set = ["R", "P", "S"]

    def __init__(self, action_sequence, action_sequence_2, sequence_change_idx, length_measurement=1):
        """Create game

        Args:
            action_sequence: sequence of actions for player 2
        """
        self.action_sequence = action_sequence
        self.action_sequence_2 = action_sequence_2
        self.sequence_change_idx = sequence_change_idx
        self.time_counter = 0
        self.last_action = len(self.action_sequence) - 1
        # create outputs with all combinations of actions
        measurement_set = self.action_set
        for i in range(length_measurement - 1):
            new_measurement_set = set()
            for m in measurement_set:
                for a in self.action_set:
                    new_measurement_set.add(m + a)
            measurement_set = new_measurement_set
        print("Measurement set with {:d} elements".format(len(measurement_set)))
        if len(measurement_set) < 28:
            print(measurement_set)
        self.measurement_set = list(measurement_set)
        # initialize measurement
        self.measurement = length_measurement * "S"

    def cost(self, p1_action, p2_action) -> float:
        """Returns game outcome

        Args:
            p1_action: action for player 1
            p2_action: action for player 2

        Returns:
            game result (float): -1 if player 1 wins, +1 if player 1 loses, 0 draw
        """
        if p1_action == p2_action:
            # draw
            return 0
        if (p1_action == "R" and p2_action == "S") or (p1_action == "P" and p2_action == "R") or (
                p1_action == "S" and p2_action == "P"):
            return -1
        else:
            return +1

    def get_measurement(self):
        """get measurement for next game

        Returns:
            measurement: measurement for next game
        """
        return self.measurement, None

    def play(self, p1_action) -> tuple[float, dict[str, float]]:
        """Play game

        Args:
            p1_action: action for player 1

        Returns: 
            tuple[float,dict(str,float)]
            cost (float): -1 if player 1 wins, +1 if player 1 loses, 0 draw
            all_costs (dict[str,float]): dictionary with costs for all actions of player 1
        """
        # select action for player 2
        self.time_counter += 1
        self.last_action += 1
        if self.last_action >= len(self.action_sequence):
            self.last_action = 0
        if self.time_counter < self.sequence_change_idx:
            p2_action = self.action_sequence[self.last_action]
        else:
            p2_action = self.action_sequence_2[self.last_action]
        # update measurement
        self.measurement = self.measurement[1:] + p2_action
        # cost for given action of player 1
        cost = self.cost(p1_action, p2_action)
        # costs for all actions of player 1
        all_costs = OrderedDict([(a, self.cost(a, p2_action)) for a in self.action_set])
        return cost, all_costs, p2_action


def empirical_sequence_analysis(initial_sequence, ax=None):
    sequences = dict()
    extended_sequence = np.hstack((initial_sequence, initial_sequence))
    for i in range(len(initial_sequence)):
        arr = extended_sequence[i:i + length_measurement]
        next_el = extended_sequence[i + length_measurement]
        next_el = label_to_action[next_el] # switch to optimal action instead of opponent action
        temp = ''
        for ar in arr:
            temp += ar
        if temp not in sequences.keys():
            sequences[temp] = []
        sequences[temp].append(next_el)

    for key, val in sequences.items():
        counts = np.zeros((3, 1))
        for i, move in enumerate(['R', 'P', 'S']):
            counts[i] = val.count(move)
        norm_counts = counts / counts.sum()
        sequences[key] = norm_counts

    freq_values = np.array(list(sequences.values())).squeeze()
    freq_values.sort(axis=1)

    sorted_array = freq_values[np.argsort(-freq_values.max(axis=1))]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Create stacked bar plot for each row
    for i in range(sorted_array.shape[0]):
        x_pos = i / len(sorted_array)
        ax.bar(x_pos, sorted_array[i, 2], color='b')
        ax.bar(x_pos, sorted_array[i, 1], bottom=sorted_array[i, 2], color='orange')
        ax.bar(x_pos, sorted_array[i, 0], bottom=sorted_array[i, 0] + sorted_array[i, 1], color='g')
    # Customize plot
    # ax.set_xlabel("Length 5 sequences ordered by uniqueness")
    ax.set_ylabel("Uniqueness percentage")
    # plt.legend(["Column 1", "Column 2", "Column 3"], loc="upper right")
    ax.grid(True, axis='y')
    ax.set_xlim((0, 1))
    return sequences


if __name__ == '__main__':
    import itertools
    from sklearn import svm, neural_network
    from benchmark_methods import MultiArmBandit, BayesianEstimator, SklearnModel, OptimalPolicy
    from lstm import StreamingLSTM
    from utils import GamePlay
    from LearningGames import LearningGame
    import pickle

    M = 10_000
    length_measurement = 5
    measurement_to_label = False
    switch_time = 5_000
    label_to_action = {'R': 'P', 'P': 'S', 'S': 'R'}
    rng = np.random.default_rng(11)
    action_sequence = rng.permutation(["R", "P", "S"] * 50)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.permutation(["R", "P", "S"] * 50)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    opt_policy1 = empirical_sequence_analysis(action_sequence, ax=ax[0])
    opt_policy2 = empirical_sequence_analysis(action_sequence2, ax=ax[1])
    ax[0].set_ylabel('Player 2 initial \n sequence uniqueness')
    ax[1].set_ylabel('Player 2 second \n sequence uniqueness')
    ax[1].set_xlabel('Percentage of 5 length sequences')
    plt.tight_layout()
    plt.show()

    print("action sequence :", action_sequence)
    game = RPS_vs_bad_rng(action_sequence=action_sequence, length_measurement=length_measurement,
                          action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    opt_pol = OptimalPolicy(opt_policy1=opt_policy1, opt_policy2=opt_policy2,
                            switch_time=switch_time, action_set=game.action_set,  # a little bit brittle how action set is defined
                            label_to_action=label_to_action)

    bayesian = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    methods = [bayesian]
    for b, l in itertools.product([1e-2, 1e-1, 1e0, 1e1], [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0.]):
    # for b, l in itertools.product([1e4], [0.]):
        lg = LearningGame(game.action_set, measurement_set=game.measurement_set,
                            decay_rate=l, inverse_temperature=b, seed=0)
        lg.reset()
        methods.append(lg)

    # mab = MultiArmBandit(action_set=game.action_set, method='epsilon_greedy', method_args=dict(epsilon=0.5))
    data_window = 1_000
    update_freq = 1_000
    svm = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
               raw_measurement=False, measurement_to_label=measurement_to_label, finite_measurement=True, policy_map=label_to_action,
              update_frequency=update_freq, model=svm.SVR(kernel='rbf'))
    nn = neural_network.MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(30, 30))
    mlp = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
               raw_measurement=False, measurement_to_label=measurement_to_label, finite_measurement=True, policy_map=label_to_action,
              update_frequency=update_freq, model=nn)
    methods.append(svm)
    methods.append(mlp)
    # methods.append(opt_pol)
    methods.reverse()

    gp = GamePlay(decision_makers=methods,
                  game=game,
                  horizon=M,
                  disp_results_per_iter=int(M/10),
                  binary_cont_measurement=False,
                  store_energy_hist=False)
    # TODO: should expose the raw measurements to the decision algorithms in principle
    outcomes = gp.play_games()

    with open('../data/rps_{}_{}.pkl'.format(M, str(measurement_to_label)), 'wb') as f:
        pickle.dump((outcomes, methods), f)
