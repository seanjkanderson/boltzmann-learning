import numpy as np
from collections import OrderedDict


class RPSVsBadRNG:
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


if __name__ == '__main__':
    from sklearn import svm, neural_network
    from examples.benchmark_methods import BayesianEstimator, SklearnModel, PrescientBayes
    from examples.simulation_utils.utils import GamePlay
    from learning_games import LearningGame
    import pickle

    M: int = 103_000  # the total number of rounds to play the game
    length_measurement: int = 5  #
    switch_time = 20_000
    beta_values = [1e-2, 1e-1, 1e0, 1e1]
    lambda_values = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0.]
    label_to_action = {'R': 'P', 'P': 'S', 'S': 'R'}
    rng = np.random.default_rng(11)
    action_sequence = rng.permutation(["R", "P", "S"] * 50)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.permutation(["R", "P", "S"] * 50)

    # benchmark model parameters
    # whether to learn a map from measurements to label (True) or estimate the cost associated with each action (False)
    measurement_to_label: bool = False
    # the window size H of recent data
    data_window: int = 1_000
    # the frequency at which the model is updated. More updates requires more time
    update_freq: int = 1_000
    # the mapping from labels to actions when learning measurement_to_label
    label_action_policy = {0: 0, 1: 1}
    # hidden layer sizes for NN/MLP
    hidden_layer_sizes = (30, 30)
    # the max number of training iterations for the NN
    max_train_iter = 500
    # kernel choice for the SVM
    svm_kernel = 'rbf'
    # random state for the sklearn models
    random_state = 1

    print("action sequence :", action_sequence)
    game = RPSVsBadRNG(action_sequence=action_sequence, length_measurement=length_measurement,
                          action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    bayesian1 = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    bayesian2 = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    opt_pol = PrescientBayes(bayes_estimator_1=bayesian1, bayes_estimator_2=bayesian2,
                            switch_time=switch_time, finish_time=M)

    bayesian = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    methods = [bayesian]
    # for b, l in itertools.product(beta_values, lambda_values):
    for b, l in zip(beta_values, lambda_values):
        lg = LearningGame(game.action_set, measurement_set=game.measurement_set,
                            decay_rate=l, inverse_temperature=b, seed=0)
        lg.reset()
        methods.append(lg)

    if measurement_to_label:
        nn_model = neural_network.MLPClassifier(random_state=random_state,
                                                max_iter=max_train_iter,
                                                hidden_layer_sizes=hidden_layer_sizes)
        svm_model = svm.SVC(kernel=svm_kernel)
    else:
        nn_model = neural_network.MLPRegressor(random_state=random_state,
                                               max_iter=max_train_iter,
                                               hidden_layer_sizes=hidden_layer_sizes)
        svm_model = svm.SVC(kernel='rbf')
    svm = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
               raw_measurement=False, measurement_to_label=measurement_to_label, finite_measurement=True, policy_map=label_to_action,
              update_frequency=update_freq, model=svm_model)
    mlp = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
               raw_measurement=False, measurement_to_label=measurement_to_label, finite_measurement=True, policy_map=label_to_action,
              update_frequency=update_freq, model=nn_model)
    methods.append(svm)
    methods.append(mlp)
    methods.append(opt_pol)
    methods.append(opt_pol)
    methods.reverse()

    gp = GamePlay(decision_makers=methods,
                  game=game,
                  horizon=M,
                  disp_results_per_iter=int(M/10),
                  binary_cont_measurement=False,
                  store_energy_hist=False)
    outcomes = gp.play_games()

    with open('../data/rps_{}_{}.pkl'.format(M, str(measurement_to_label)), 'wb') as f:
        pickle.dump((outcomes, methods), f)
