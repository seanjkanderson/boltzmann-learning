import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import LearningGames


class RPS_vs_bad_rng:
    action_set = ["R", "P", "S"]

    def __init__(self, action_sequence=["R", "P", "S"], length_measurement=1):
        """Create game

        Args:
            action_sequence: sequence of actions for player 2
        """
        self.action_sequence = action_sequence
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
        return self.measurement

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
        self.last_action += 1
        if self.last_action >= len(self.action_sequence):
            self.last_action = 0
        p2_action = self.action_sequence[self.last_action]
        # update measurement
        self.measurement = self.measurement[1:] + p2_action
        # cost for given action of player 1
        cost = self.cost(p1_action, p2_action)
        # costs for all actions of player 1
        all_costs = OrderedDict([(a, self.cost(a, p2_action)) for a in self.action_set])
        return cost, all_costs, p2_action


if __name__ == '__main__':
    from benchmark_methods import MultiArmBandit, MultiArmBanditFullInfo, SVMRegressor
    from lstm import StreamingLSTM
    from utils import GamePlay

    rng = np.random.default_rng(11)
    action_sequence = rng.permutation(["R", "P", "S"] * 50)
    print("action sequence :", action_sequence)
    game = RPS_vs_bad_rng(action_sequence=action_sequence, length_measurement=5)
    mab_game = RPS_vs_bad_rng(action_sequence=action_sequence, length_measurement=5)
    lg = LearningGames.LearningGame(game.action_set, measurement_set=game.measurement_set,
                                    decay_rate=1e-5, inverse_temperature=.1, seed=0)

    lg.reset()
    mab = MultiArmBandit(action_set=game.action_set, method='epsilon_greedy', method_args=dict(epsilon=0.5))
    mab_full = MultiArmBanditFullInfo(action_set=game.action_set, measurement_set=game.measurement_set)
    svm = SVMRegressor(window_size=300, kernel='rbf', action_set=game.action_set, measurement_set=game.measurement_set)
    lstm = StreamingLSTM(action_set=game.action_set, measurement_set=game.measurement_set,
                         lstm_units=100, window_size=10000, finite_measurement=True)
    M = 100_000

    gp = GamePlay(decision_makers=[lstm, lg, mab, mab_full, svm],
                  game=game,
                  horizon=M,
                  disp_results_per_iter=M+1,
                  binary_cont_measurement=False)

    outcomes = gp.play_games()

    gp.plot_game_outcome(outcomes)


    # costs = np.zeros(M)
    # mab_costs = np.zeros(M)
    # cost_bounds = np.zeros(M)
    # entropy = np.zeros(M)
    # for iter in range(M):
    #     # Play
    #     measurement = game.get_measurement()
    #     (action, _, entropy[iter]) = lg.get_action(measurement, iter)
    #     (costs[iter], all_costs) = game.play(action)
    #     measurement_idx = game.measurement_set.index(measurement)
    #     mab_action= mab.get_action(idx=iter, measurement=measurement)
    #     (mab_costs[iter], all_mab_costs) = mab_game.play(mab_action)
    #     # Learn
    #     lg.update_energies(measurement, all_costs, iter)
    #     # all_mab_costs = np.array([all_mab_costs[key] for key in game.action_set])  # assumes action set is ordered
    #     mab.update_energies(measurement=measurement,
    #                         action_cost=mab_costs[iter], costs=all_mab_costs)
    #     # Store regret
    #     (_, _, cost_bounds[iter], _, _, _, _, _) = lg.get_regret(display=False)
    #     # Output
    #     if iter < 10:
    #         print("iter={:4d}, measurement = {:s}, action = {:s}, cost = {:2.0f}, all_costs = {:s}".format(iter,
    #                                                                                                        measurement,
    #                                                                                                        action,
    #                                                                                                        costs[iter],
    #                                                                                                        str(all_costs)))
    #     # print(lg.energy)
    # lg.get_regret(display=True)
    #
    # iters = range(M)
    # average_costs = np.divide(np.cumsum(costs), np.add(range(M), 1))
    # average_mab_costs = np.divide(np.cumsum(mab_costs), np.add(range(M), 1))
    # average_cost_bounds = np.divide(np.cumsum(cost_bounds), np.add(range(M), 1))
    #
    # plot_simulation_results(iters, costs, average_costs, [entropy], average_cost_bounds=None, title_prefix="Bad RNG")
    # plot_simulation_results(iters, mab_costs, average_mab_costs, [entropy], average_cost_bounds=None, title_prefix="MAB | Bad RNG")
