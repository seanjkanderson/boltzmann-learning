import itertools

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
import lightgbm as lgb
import pickle

from src.utils import plot_simulation_results
from src.dataset_game import DatasetGame


class BinaryClassifierGame(DatasetGame):

    def __init__(self, action_set: list, measurement_set: list, action_sequence,
                 measurement_sequence, finite_measurements: bool, type1_weight=1., type2_weight=1.):
        """Create game. The goal is to minimize the cost := type1_weight*p(false pos) + type2_weight*p(false neg)

        Args:
            type1_weight (float): the weight for false positive rates
            type2_weight (float): the weight for false negative rates
        """
        super().__init__(action_set=action_set, measurement_set=measurement_set, action_sequence=action_sequence,
                         measurement_sequence=measurement_sequence, finite_measurements=finite_measurements)
        self.type1_weight = type1_weight
        self.type2_weight = type2_weight

    def cost(self, p1_action, p2_action) -> float:
        """Returns game outcome

        Args:
            p1_action: action for player 1
            p2_action: action for player 2

        Returns:
            game result (float): A*false_pos + B*false_neg
        """
        # while in principle we care about the probability computed over time, the cost can be written as a stage cost
        # as total_cost^T = A*\sum_{t=0}^T I(a_1^t=1, a_2^t=0)/T + B*\sum_{t=0}^T I(a_1^t=0, a_2^t=1)/T
        #               = A*\sum_{t=0}^{T-1} I(a_1^t=1, a_2^t=0)/T + B*\sum_{t=0}^{T-1} I(a_1^t=0, a_2^t=1)/T +
        #                 A*I(a_1^T=1, a_2^T=0)/T + B*I(a_1^T=0, a_2^T=1)/T
        #               = A*I(a_1^T=1, a_2^T=0)/T + B*I(a_1^T=0, a_2^T=1)/T + total_cost^{T-1}*(T-1)/T
        false_pos = (p1_action == 1) and (p2_action == 0)
        false_neg = (p1_action == 0) and (p2_action == 1)
        return self.type1_weight * false_pos + self.type2_weight * false_neg


def classifier(train_length: int, test_length_first: int, test_length_second: int, rng: np.random, type1_weight, type2_weight):

    def generate_only_or(n):
        x = np.zeros((n, 2))
        x[:n // 3, 0] = 0
        x[n // 3: 2*n // 3, 0] = 0
        x[2*n // 3:, 0] = 1
        x[:n // 3, 1] = 1
        x[n // 3: 2 * n // 3, 1] = 0
        x[2*n // 3:, 1] = 0
        y = np.logical_xor(x[:, 0], x[:, 1]).astype(int).reshape(-1, 1)

        xy = np.hstack((x, y))
        np.random.shuffle(xy)
        return xy[:, :2], xy[:, 2, np.newaxis]

    def generate_labeled_data():
        x_train, y_train = generate_only_or(train_length)
        x_first, y_first = generate_only_or(test_length_first)

        x_second = rng.permutation([0, 1] * test_length_second).reshape(-1, 2)
        y_second = np.logical_xor(x_second[:, 0], x_second[:, 1]).astype(int).reshape(-1, 1)

        x_test = np.vstack((x_first, x_second))
        y_test = np.vstack((y_first, y_second))

        return x_train, y_train, x_test, y_test

    def class_assignment(y_pred_prob: float, threshold: float):
        return (y_pred_prob >= threshold).astype(int)


    x_train, y_train, x_test, p2_act_sequence = generate_labeled_data()

    mdl = GradientBoostingClassifier(n_estimators=100, learning_rate=.1,
    max_depth=1, random_state=0)
    mdl.fit(x_train, y_train)
    train_probs = mdl.predict_proba(x_train)
    measurement_sequence = mdl.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_train, train_probs[:, 0])
    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    # TODO: finish assignment
    # optimal_threshold = find_optimal_threshold(y_train, train_probs[:, 1], type1_weight, type2_weight)
    classifier_predictions = class_assignment(y_pred_prob=measurement_sequence[:, 1], threshold=0.5)

    return measurement_sequence, p2_act_sequence, classifier_predictions


def find_optimal_threshold(y_true, y_pred_proba, A, B):
    thresholds = np.linspace(0, 1, 10000)
    best_threshold = thresholds[0]
    min_cost = float('inf')

    for T in thresholds:
        false_positive_rate = np.mean((y_pred_proba >= T) & (y_true == 0))
        false_negative_rate = np.mean((y_pred_proba < T) & (y_true == 1))

        total_cost = A * false_positive_rate + B * false_negative_rate
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = T
    print('Optimal threshold={:.2f} for error weights I:{} II:{}'.format(best_threshold, A, B))
    return best_threshold


def ember_classifier(n_train, n_features, type1_weight, type2_weight):

    data = dict(x_train=np.array([]), x_test=np.array([]), y_train=np.array([]), y_test=np.array([]))
    for key, _ in data.items():
        with open('../ember_np/{}.pkl'.format(key), 'rb') as f:
            data[key] = pickle.load(f)

    # only include the labeled points
    labeled = data['y_train'] >= 0.
    labeled_test = data['y_test'] >= 0.
    x_train = data['x_train'][labeled]
    x_test = data['x_test'][labeled_test]
    y_train = data['y_train'][labeled]
    y_test = data['y_test'][labeled_test]

    x_train_sub = x_train[:n_train, :n_features]
    x_test_sub = x_test[:n_train, :n_features]
    y_train_sub = y_train[:n_train]
    y_test_sub = y_test[:n_train]

    train = lgb.Dataset(x_train_sub, y_train_sub)
    lgbm_model = lgb.train(dict(num_leaves=31, objective='binary'), train)

    y_pred_proba = lgbm_model.predict(x_train_sub)
    measurement_sequence = lgbm_model.predict(x_test_sub)
    p2_act_sequence = y_test_sub

    # classify based on a threshold that takes into account type1 and type2 errors
    threshold = find_optimal_threshold(y_train_sub, y_pred_proba, type1_weight, type2_weight)
    classifier_predictions = (measurement_sequence > threshold).astype(int)

    # need to include probabilities of all classes
    measurement_sequence = np.array([1. - measurement_sequence, measurement_sequence]).T
    measurement_set = [0, 1]
    return measurement_sequence, p2_act_sequence, classifier_predictions, threshold, measurement_set


def toy_classifier(n_points, type1_weight, type2_weight, finite_measurements: bool):

    # create a dataset where the classifier's probability is uniformly sampled on [0, 1]
    measurement_sequence = np.random.uniform(size=n_points)
    # top_sequence = measurement_sequence[measurement_sequence > .75]
    # bottom_sequence = measurement_sequence[measurement_sequence < .25]
    # measurement_sequence = np.hstack((top_sequence, bottom_sequence))

    # measurement_sequence.sort()
    # measurement_sequence = measurement_sequence[::-1]
    # then select whether each point is malicious or not with equal probability to the malicious probability
    random_values = np.random.uniform(size=len(measurement_sequence))

    # Sample points according to probs
    p2_act_sequence = (random_values < measurement_sequence).astype(int)

    threshold = find_optimal_threshold(p2_act_sequence, measurement_sequence, type1_weight, type2_weight)
    classifier_predictions = (measurement_sequence > threshold).astype(int)

    if finite_measurements:
        measurement_sequence = classifier_predictions
    else:
        measurement_sequence = np.array([1-measurement_sequence, measurement_sequence]).T

    measurement_set = [0, 1]
    return measurement_sequence, p2_act_sequence, classifier_predictions, threshold, \
        len(measurement_sequence), measurement_set


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from src.LearningGames import LearningGame

    finite_meas = False
    M = 100_000
    m_iter = int(M/10)
    type1 = 1.
    type2 = 1.
    def_rng = np.random.default_rng(11)
    # p2_act_sequence = rng.permutation(["R", "P", "S"] * 50)
    # meas_sequence, p2_act_sequence, classifier_pred = classifier(train_length=10,
    #                                                              test_length_first=10,
    #                                                              test_length_second=M + 1,
    #                                                              rng=def_rng,
    #                                                              type1_weight=1.,
    #                                                              type2_weight=2.)
    # meas_sequence, p2_act_sequence, classifier_action, classifier_threshold, outcomes_set = ember_classifier(n_train=100000,
    #                                                                                            n_features=100,
    #                                                                      type1_weight=1., type2_weight=1.)
    meas_sequence, p2_act_sequence, classifier_action, classifier_threshold, M, outcomes_set = toy_classifier(M,
                                                                                             type1_weight=type1,
                                                                                             type2_weight=type2,
                                                                                             finite_measurements=finite_meas)

    print('Measurement set: {}'.format(str(outcomes_set)))
    game = BinaryClassifierGame(action_sequence=p2_act_sequence, measurement_sequence=meas_sequence, measurement_set=outcomes_set,
                       action_set=[0, 1], type1_weight=type1, type2_weight=type2, finite_measurements=finite_meas)
    lg = LearningGame(game.action_set, measurement_set=game.measurement_set, finite_measurements=finite_meas,
                                    decay_rate=0., inverse_temperature=1e-2, seed=0)

    lg.reset()

    costs = np.zeros(M)
    cost_bounds = np.zeros(M)
    entropy = np.zeros(M)
    classifier_cost = []
    p1_action = []
    tt = 0
    tt_one = 0
    for idx in range(M):
        # Play
        measurement = game.get_measurement()
        (action, prob, entropy[idx]) = lg.get_action(measurement, idx)
        (costs[idx], all_costs) = game.play(action)
        p1_action.append(action)
        # Learn
        lg.update_energies(measurement, all_costs, idx)
        # Store regret
        (_, _, cost_bounds[idx], _, _, _, _, _) = lg.get_regret(display=False)
        # Get cost of using classifier
        classifier_cost.append(game.cost(classifier_action[idx], p2_act_sequence[idx]))
        # Output
        if (idx % m_iter == 0) or (idx == M - 1):
            print("iter={:4d}, action_1 = {:2.0f}, action_2 = {:2.0f}, cost = {:2.0f}, all_costs = {:s}".format(idx,
                                                                                                            action,
                                                                                                            p2_act_sequence[idx],
                                                                                                            costs[idx],
                                                                                                            str(all_costs)))
            print('measurement: {}'.format(str(measurement)))
            # print("previous mal prob: {:.3f} | updated mal prob: {:.3f}".format(prob[1], prob_update[1]))
            print("energy: {}".format(lg.energy))
            fig, ax = plt.subplots(2, 1, sharex='col')
            p = []
            meas_range = np.arange(0.1, 1.0, step=.01).round(2)
            pos_res = []
            bin_counts = []
            # TODO: generalize the below code. Plot the optimal policy as function of measurements
            #      Plot the FPR and FNR in ROC?
            meas_seq_probs = meas_sequence[:, 1].round(2)
            for measurement_m in meas_range:
                (_, prob_m, _) = lg.get_action(measurement={0: 1 - measurement_m, 1: measurement_m}, time=idx)
                p.append(prob_m[1])
                instance_count = (meas_seq_probs == measurement_m).sum()
                bin_counts.append(instance_count)
                pos_res.append(np.sum(p2_act_sequence[meas_seq_probs == measurement_m] == 1) / instance_count)
            p = np.array(p)
            pos_res = np.array(pos_res)

            ax[0].plot(meas_range, p, label='Boltzmann')
            # ax[0].axvline(x=classifier_threshold, c='red', label='classifier threshold')
            classifier_policy = (meas_range > classifier_threshold).astype(int)
            ax[0].scatter(meas_range, classifier_policy, c='red', label='classifier policy')
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

        # print(lg.energy)
    lg.get_regret(display=True)

    # print('total_cost:', (type1*(np.array(p1_action) > p2_act_sequence).sum() + type2 * (
    #         np.array(p1_action) < p2_act_sequence).sum()) / M)
    # print('total_cost_classifier:', (type1 * (np.array(classifier_action) > p2_act_sequence).sum() + type2 * (
    #             np.array(classifier_action) < p2_act_sequence).sum()) / M)

    iters = range(M)
    average_costs = np.divide(np.cumsum(costs), range(1, M + 1))
    average_costs_classifier = np.divide(np.cumsum(classifier_cost), range(1, M + 1))
    average_cost_bounds = np.divide(np.cumsum(cost_bounds), np.add(range(M), 1))
    print('Average costs: {} | Average classifier cost: {}'.format(average_costs[-10:].mean(), average_costs_classifier[-100:].mean()))
    plot_simulation_results(iters, costs, average_costs, [entropy], average_cost_bounds=None, title_prefix="Dataset Game")
