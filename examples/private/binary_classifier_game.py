import numpy as np

from examples.simulation_utils.dataset_game import DatasetGame


class BinaryClassifierGame(DatasetGame):

    def __init__(self, action_set: list, measurement_set: list, opponent_action_sequence,
                 measurement_sequence, finite_measurements: bool, type1_weight=1., type2_weight=1.):
        """Create game. The goal is to minimize the cost := type1_weight*p(false pos) + type2_weight*p(false neg)

        Args:
            type1_weight (float): the weight for false positive rates
            type2_weight (float): the weight for false negative rates
        """
        super().__init__(action_set=action_set, measurement_set=measurement_set,
                         opponent_action_sequence=opponent_action_sequence,
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


def toy_classifier(n_points, type1_weight, type2_weight, finite_measurements: bool, case: int):

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

    # threshold = find_optimal_threshold(p2_act_sequence, measurement_sequence, type1_weight, type2_weight)
    threshold = type1_weight / (type1_weight + type2_weight)
    threshold_info = dict(threshold=threshold,
                          cost=type1_weight/2 - type1_weight*threshold + threshold**2/2*(type1_weight + type2_weight))
    classifier_predictions = (measurement_sequence > threshold).astype(int)

    if finite_measurements:
        measurement_sequence = classifier_predictions
    else:
        if case == 1:
            measurement_sequence = np.array([1-measurement_sequence, measurement_sequence]).T
            measurement_set = [0, 1]
        elif case == 2:
            measurement_sequence, measurement_set = phi(measurement_sequence)

    return measurement_sequence, p2_act_sequence, classifier_predictions, threshold_info, \
        max(measurement_sequence.shape), measurement_set


def phi(measurement_sequence):
    thresholds = np.arange(0, 1, 0.05)
    gamma = 50.
    measurement_set = [str(np.round(tau, 2)) for tau in thresholds]
    threshold_classes = [np.exp(-gamma * np.abs(measurement_sequence - tau)) for tau in thresholds]
    measurement_sequence = np.vstack(threshold_classes).T
    measurement_sequence = measurement_sequence / measurement_sequence.sum(axis=1)[..., np.newaxis]
    return measurement_sequence, measurement_set


def nonstationary_classifier(n_points, type1_weight, type2_weight, finite_measurements: bool, case: int):

    # create a dataset where the classifier's probability is uniformly sampled on [0, 1]
    measurement_sequence = np.random.uniform(size=n_points)
    random_values = np.random.uniform(size=len(measurement_sequence))

    # Sample points according to probs but with a positive bias after the first half
    e_vec = np.zeros((n_points,))
    e_vec[-int(n_points/2):] = 1.
    actual_probs = measurement_sequence - e_vec*0.2 # np.random.uniform(low=0., high=0.3, size=n_points)*e_vec
    actual_probs[actual_probs > 1.] = 1.
    p2_act_sequence = (random_values < actual_probs).astype(int)

    # threshold = find_optimal_threshold(p2_act_sequence, measurement_sequence, type1_weight, type2_weight)
    threshold = type1_weight / (type1_weight + type2_weight)
    threshold_info = dict(threshold=threshold,
                          cost=type1_weight / 2 - type1_weight * threshold + threshold ** 2 / 2 * (
                                      type1_weight + type2_weight))
    classifier_predictions = (measurement_sequence > threshold).astype(int)

    if finite_measurements:
        measurement_sequence = classifier_predictions
    else:
        if case == 1:
            measurement_sequence = np.array([1-measurement_sequence, measurement_sequence]).T
            measurement_set = [0, 1]
        elif case == 2:
            measurement_sequence, measurement_set = phi(measurement_sequence)

    return measurement_sequence, p2_act_sequence, classifier_predictions, threshold_info, \
        max(measurement_sequence.shape), measurement_set


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from examples.simulation_utils.utils import plot_simulation_results
    from src.learning_games import LearningGame

    finite_meas = False
    meas_case = 1
    M = 150_000
    m_iter = int(M/10)
    type1 = 1.
    type2 = 1.5
    def_rng = np.random.default_rng(11)
    kernel_rbf = lambda x, y: np.exp(-30*np.linalg.norm(x - y, 2, axis=-1))
    kernel_poly = lambda x, y: (x.T@y + .1)**2
    # kernel_list = [None, None, kernel_rbf]
    kernel_list = [None]
    data = []
    # for case_name, meas_case, beta, kernel in zip(['nominal', 'lifted', 'kernel'], [1, 2, 1], [1e-4, 1e-2, 1e-4], kernel_list):
    for case_name, meas_case, beta, kernel in zip(['nominal'], [1], [1e-3], kernel_list):
        print(meas_case, kernel)
        # meas_sequence, p2_act_sequence, classifier_action, \
        #     threshold_info, M, outcomes_set = toy_classifier(M,
        #                                                          type1_weight=type1,
        #                                                          type2_weight=type2,
        #                                                          finite_measurements=finite_meas,
        #                                                           case=meas_case)
        meas_sequence, p2_act_sequence, classifier_action, \
            threshold_info, M, outcomes_set = nonstationary_classifier(M,
                                                                              type1_weight=type1,
                                                                              type2_weight=type2,
                                                                              finite_measurements=finite_meas,
                                                                              case=meas_case)

        print('Measurement set: {}'.format(str(outcomes_set)))
        game = BinaryClassifierGame(opponent_action_sequence=p2_act_sequence,
                                    measurement_sequence=meas_sequence, measurement_set=outcomes_set,
                                    action_set=[0, 1], type1_weight=type1,
                                    type2_weight=type2, finite_measurements=finite_meas)
        lg = LearningGame(game.action_set, measurement_set=game.measurement_set, finite_measurements=finite_meas,
                                        decay_rate=0.*1e-6, inverse_temperature=beta, seed=0, kernel=kernel)

        lg.reset()

        costs = np.zeros(M)
        cost_bounds = np.zeros(M)
        entropy = np.zeros(M)
        classifier_cost = []
        p1_action = []
        tt = 0
        tt_one = 0
        for idx in range(M-1):
            # Play
            measurement, _ = game.get_measurement()
            (action, prob, entropy[idx]) = lg.get_action(measurement, idx)
            (costs[idx], all_costs, _) = game.play(action)
            p1_action.append(action)
            # Learn
            lg.update_energies(measurement, all_costs, idx)
            # Store regret
            (_, _, cost_bounds[idx], _, _, _, _, _) = lg.get_regret(display=False)
            # Get cost of using classifier
            classifier_cost.append(game.cost(classifier_action[idx], p2_act_sequence[idx]))
            # Output
            if (idx % m_iter == 0) or (idx == M - 1):
                print("iter={:4d}, action_1 = {:2.0f}, action_2 = {:2.0f},"
                      " cost = {:2.0f}, all_costs = {:s}".format(idx,
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
                    if meas_case == 2:
                        meas_arr, meas_set = phi(measurement_m)
                        meas = {key: val[0] for val, key in zip(meas_arr.T, meas_set)}
                    else:
                        meas = {0: 1 - measurement_m, 1: measurement_m}
                    (_, prob_m, _) = lg.get_action(measurement=meas, time=idx)
                    p.append(prob_m[1])
                    instance_count = (meas_seq_probs == measurement_m).sum()
                    bin_counts.append(instance_count)
                    pos_res.append(np.sum(p2_act_sequence[meas_seq_probs == measurement_m] == 1) / instance_count)
                p = np.array(p)
                pos_res = np.array(pos_res)

                ax[0].plot(meas_range, p, label='Boltzmann')
                # ax[0].axvline(x=classifier_threshold, c='red', label='classifier threshold')
                classifier_policy = (meas_range > threshold_info['threshold']).astype(int)
                ax[0].step(meas_range, classifier_policy, c='red', label='classifier policy')
                # ax[0].plot(meas_range, pos_res, label='empirical % malicious')
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
        average_costs_classifier = np.divide(np.cumsum(classifier_cost), range(1, M))
        average_cost_bounds = np.divide(np.cumsum(cost_bounds), np.add(range(M), 1))
        print('Average costs: {} | Average classifier cost: {}'.
              format(average_costs[-10:].mean(), average_costs_classifier[-100:].mean()))
        plot_simulation_results(iters, costs, average_costs, [entropy], average_cost_bounds=None, title_prefix="Dataset Game")

        data.append([meas_range, p, costs, average_costs, average_cost_bounds, case_name])

    plt.clf()
    fig3, ax3 = plt.subplots(2)
    classifier_policy = (meas_range > threshold_info['threshold']).astype(int)
    ax3[0].step(meas_range, classifier_policy, c='red', label='classifier policy')

    ax3[1].axhline(y=threshold_info['cost'], label='opt threshold', linestyle='-.', c='g')
    # ax3[1].axhline(y=1/3, label='opt linear', linestyle='-.', c='red') # TODO: better not to hardcode this and use optimal cost formula
    iters_scaled = [i/1000 for i in iters]
    for (meas_range, p, costs, average_costs, average_cost_bounds, case_name) in data:
        print(case_name)

        ax3[0].plot(meas_range, p, label=case_name)
        # ax[0].axvline(x=classifier_threshold, c='red', label='classifier threshold')

        # ax[0].plot(meas_range, pos_res, label='empirical % malicious')
        ax3[0].legend()
        ax3[0].set_ylabel('Prob a=1')
        ax3[0].set_xlabel('Measurement prob l=1')
        ax3[0].set_ylim(0, 1.1)
        ax3[0].set_xlim(0, 1.)

        # lines = ax3[1].plot(iters, costs, '.', color='tab:blue')
        lines = ax3[1].plot(iters_scaled, average_costs, '-', label=case_name)
        # if average_cost_bounds is not None:
        #     lines += ax3[1].plot(iters, average_cost_bounds, '-', color='tab:cyan')
        # for idx, entropy in enumerate(entropies):
        #     lines += ax_twin.plot(iters, entropy, '.', label='entropy: P{}'.format(idx+1), color='tab:green')

        ax3[1].set_xlabel(r't  ($x10^3$)')
        ax3[1].set_ylabel('average cost')
        ax3[1].set_ylim((threshold_info['cost']*.95, .6))
        ax3[1].legend(bbox_to_anchor=(1., 0.75))

    ax3[0].grid()
    ax3[1].grid()
    plt.tight_layout()
    plt.show()