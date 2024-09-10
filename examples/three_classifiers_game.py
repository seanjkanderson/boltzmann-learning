import itertools

import numpy as np
from scipy.stats import norm, multivariate_normal

from dataset_game import DatasetGame
from utils import generate_probabilities_matrix


class ThreeClassifiersGame(DatasetGame):

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
        return self.type1_weight*false_pos + self.type2_weight*false_neg


def three_classifiers(n_points: int, finite_measurements: bool, case: int, measurement_case: int):
    """
    Generates a dataset based on three classifiers
    Args:
        n_points: the number of datapoints to generate for the game (>= to game length)
        finite_measurements: whether to use a finite set of measurements or infinite
        case: 1: first classifier is always right, the others are random
                2: first classifier gets the probabilities right, the second flips them, and third is random
                3: first classifier gets the probabilities right, the second is overconfident, third underconfident
        measurement_case:
    Returns:
        measurement_sequence (np.array): the sequence of measurements
        p2_act_sequence (np.array): player 2's sequence of actions
        classifier_prediction (np.array): a sequence of binary decisions
        outcomes (list): the set of possible measurements

    """

    if case == 1:
        # first classifier is always right, the others are random
        measurement_sequence_1 = np.random.uniform(size=n_points).round(0)
        measurement_sequence_2 = np.random.uniform(size=n_points)
        measurement_sequence_3 = np.random.uniform(size=n_points)
    elif case == 2:
        # first classifier gets the probabilities right, the second flips them, and third is random
        measurement_sequence_1 = np.random.uniform(size=n_points)
        measurement_sequence_2 = 1 - measurement_sequence_1
        measurement_sequence_3 = np.random.uniform(size=n_points)
    elif case == 3:
        # first classifier gets the probabilities right, the second is overconfident, third underconfident
        measurement_sequence_1 = np.random.uniform(size=n_points)
        measurement_sequence_2 = measurement_sequence_1 + np.random.uniform(size=n_points, low=0.0, high=0.2)
        measurement_sequence_2[measurement_sequence_2 > 1.] = 1.
        measurement_sequence_3 = measurement_sequence_1 + np.random.uniform(size=n_points, low=-0.2, high=0.)
        measurement_sequence_3[measurement_sequence_3 < 0.] = 0.
    else:
        raise ValueError('Not a valid option.')

    indep_probs = np.vstack((measurement_sequence_1, measurement_sequence_2, measurement_sequence_3)).T

    if measurement_case == 1:
        # take the classifiers to be independent and compute the probability of each outcome
        outcomes, measurement_sequence = generate_probabilities_matrix(indep_probs)
        # outcomes += ['1_1', '1_2', '1_3']
        # measurement_sequence = np.hstack((measurement_sequence, indep_probs))
        # measurement_sequence = measurement_sequence / measurement_sequence.sum(axis=1)[:, np.newaxis]

        if finite_measurements:
            measurement_sequence = np.array([(indep_probs[:, 0] > 0.5).astype(int), (indep_probs[:, 1] > 0.75).astype(int), (indep_probs[:, 2] > 0.95).astype(int)]).T
            measurement_sequence = np.array(["".join(outcome.astype(str)) for outcome in measurement_sequence])
    elif measurement_case == 2:
        # the classes are {0,1} for each classifier. Can also use softmax leading to similar results
        measurement_sequence = np.vstack((1-measurement_sequence_1, measurement_sequence_1,
                                          1-measurement_sequence_2, measurement_sequence_2,
                                          1-measurement_sequence_3, measurement_sequence_3)).T
        measurement_sequence = measurement_sequence / measurement_sequence.sum(axis=1)[..., np.newaxis]
        outcomes = ['0_1', '1_1', '0_2', '1_2', '0_3', '1_3']
    elif measurement_case == 3:
        thresholds = np.arange(0, 1, 0.1)
        threshold_classes = [(indep_probs.mean(axis=1) > tau).astype(int)*(indep_probs.mean(axis=1) - tau) for tau in thresholds]
        measurement_sequence = np.vstack(threshold_classes).T
        measurement_sequence = measurement_sequence / measurement_sequence.sum(axis=1)[..., np.newaxis]
        outcomes = [str(np.round(tau, 2)) for tau in thresholds]
    elif measurement_case == 4:
        u1 = norm.cdf(measurement_sequence_1)
        u2 = norm.cdf(measurement_sequence_2)
        u3 = norm.cdf(measurement_sequence_3)

        # Correlation matrix to model dependency (e.g., 0.5 correlation)
        rho_1_2 = 0.
        rho_2_3 = 0.
        rho_1_3 = 0.
        cov_matrix = [[1., rho_1_2, rho_1_3], [rho_1_2, 1., rho_2_3], [rho_1_3, rho_2_3, 1.]]

        # Gaussian copula
        joint_probs = multivariate_normal.cdf(np.column_stack([u1, u2, u3]), mean=[0., 0., 0.], cov=cov_matrix)

        # Marginal probabilities
        marginal_p1 = norm.pdf(measurement_sequence_1)
        marginal_p2 = norm.pdf(measurement_sequence_2)
        marginal_p3 = norm.pdf(measurement_sequence_3)

        # Compute joint probability (using Gaussian copula)
        measurement_sequence_1 = joint_probs * marginal_p1 * marginal_p2 * marginal_p3
        measurement_sequence = np.vstack((1-measurement_sequence_1, measurement_sequence_1)).T
        outcomes = ['0', '1']

    random_values = np.random.uniform(size=len(measurement_sequence_1))
    p2_act_sequence = (random_values < measurement_sequence_1).astype(int)

    # TODO: this is arbitrary and perhaps something more insightful can be done
    classifier_prediction = (indep_probs.mean(axis=1) > 0.5).astype(int)

    return measurement_sequence, p2_act_sequence, classifier_prediction, outcomes


if __name__ == '__main__':
    from src.LearningGames import LearningGame
    from src.utils import plot_simulation_results

    finite_meas = True
    M = 100_000
    m_iter = int(M/10)
    type1 = 1.
    type2 = 1.
    def_rng = np.random.default_rng(11)
    case = 2
    meas_case = 1
    kernel = lambda x, y: np.exp(-200 * np.linalg.norm(x - y, 2, axis=-1))
    kernel = None

    meas_sequence, p2_act_sequence, classifier_action, outcomes_set = three_classifiers(M,
                                                                                        finite_measurements=finite_meas,
                                                                                        case=case, measurement_case=meas_case)

    print('Measurement set: {}'.format(str(outcomes_set)))
    game = ThreeClassifiersGame(opponent_action_sequence=p2_act_sequence,
                                measurement_sequence=meas_sequence, measurement_set=outcomes_set,
                       action_set=[0, 1], type1_weight=type1, type2_weight=type2, finite_measurements=finite_meas)
    lg = LearningGame(game.action_set, measurement_set=game.measurement_set, finite_measurements=finite_meas,
                                    decay_rate=0., inverse_temperature=1e-2, seed=0, kernel=kernel)

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
            print("energy: {}".format(lg.energy))

    lg.get_regret(display=True)

    iters = range(M)
    average_costs = np.divide(np.cumsum(costs), range(1, M + 1))
    average_costs_classifier = np.divide(np.cumsum(classifier_cost), range(1, M))
    average_cost_bounds = np.divide(np.cumsum(cost_bounds), np.add(range(M), 1))
    print('Average costs: {} | Average classifier cost: {}'.format(average_costs[-10:].mean(), average_costs_classifier[-100:].mean()))

    # Test the extreme cases for possible measurements by assigning probability of one to each class
    print('Sanity check extreme cases:')
    for key in outcomes_set:
        if finite_meas:
            measurement = key
        else:
            measurement = {key: 0 for key in outcomes_set}
            measurement[key] = 1.
        (_, prob_m, _) = lg.get_action(measurement, time=idx)
        print('{} | p_0: {:0.2f}'.format(key, prob_m[0]))

    plot_simulation_results(iters, costs, average_costs, [entropy], average_cost_bounds=None, title_prefix="Dataset Game")
