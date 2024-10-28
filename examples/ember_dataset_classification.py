import itertools
import os
from collections import Counter

import numpy as np
import lightgbm as lgb
import pickle
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans

from dataset_game import DatasetGame
from utils import generate_probabilities_matrix


class EMBERClassifierGame(DatasetGame):

    def __init__(self, action_set: list, measurement_set: list, opponent_action_sequence, raw_measurements,
                 measurement_sequence, finite_measurements: bool, type1_weight=1., type2_weight=1.):
        """Create game. The goal is to minimize the cost := type1_weight*p(false pos) + type2_weight*p(false neg)

        Args:
            type1_weight (float): the weight for false positive rates
            type2_weight (float): the weight for false negative rates
        """
        super().__init__(action_set=action_set, measurement_set=measurement_set, raw_measurements=raw_measurements,
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


def custom_loss(weights_fp, weights_fn):
    def loss_function(y_true, y_pred):
        """
        Custom loss for binary classification that applies different weights to false positives and false negatives.

        Args:
        - y_true: true binary labels
        - y_pred: predicted scores (before applying sigmoid)

        Returns:
        - grad: gradient (first derivative of loss with respect to y_pred)
        - hess: hessian (second derivative of loss with respect to y_pred)
        """
        # Apply sigmoid to predictions
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))

        # Calculate gradients and hessians
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)

        # Weight false positives and false negatives differently
        # y_true == 1: false negatives
        # y_true == 0: false positives
        grad[y_true == 1] = (y_pred[y_true == 1] - 1) * weights_fn
        grad[y_true == 0] = y_pred[y_true == 0] * weights_fp

        hess[y_true == 1] = y_pred[y_true == 1] * (1 - y_pred[y_true == 1]) * weights_fn
        hess[y_true == 0] = y_pred[y_true == 0] * (1 - y_pred[y_true == 0]) * weights_fp

        return grad, hess

    return loss_function


def ember_classifier(n_train, n_test, n_features, type1_weight, type2_weight, model_file):
    data = dict(x_train=np.array([]), x_test=np.array([]), y_train=np.array([]), y_test=np.array([]))
    for key, _ in data.items():
        with open('../ember_np/{}.pkl'.format(key), 'rb') as f:
            data[key] = pickle.load(f)

    # only include the labeled points
    labeled = data['y_train'] >= 0.
    labeled_test = data['y_test'] >= 0.
    x_train = data['x_train'][labeled]
    x_test = data['x_test'][labeled_test]
    x_total = np.vstack((x_train, x_test))
    y_train = data['y_train'][labeled]
    y_test = data['y_test'][labeled_test]
    y_total = np.hstack((y_train, y_test)).T

    time_sort = x_total[:, 626].argsort()
    x_total = x_total[time_sort]
    y_total = y_total[time_sort]

    x_train = x_total[:n_train]
    x_test = x_total[n_train:n_train+n_test]

    x_train, x_scale = normalize_features(x_train)
    x_test, _ = normalize_features(x_test, x_scale)

    # reorder_idx = np.arange(x_total.shape[0])
    # np.random.shuffle(reorder_idx)
    # x_total = x_total[reorder_idx]
    # y_total = y_total[reorder_idx]

    measurement_sequences = []
    y_train = y_total[:n_train]
    y_test = y_total[n_train:n_train + n_test]
    if len(y_train) < n_train:
        raise IndexError('Not enough points in training set')
    if len(y_test) < n_test:
        raise IndexError('Not enough points in test set')

    kmeans = False
    if kmeans:
        # Kmeans
        gamma = 10
        n_clusters = 500
        outcomes = [i for i in range(n_clusters)]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(x_train)
        y_pred_train = kmeans.predict(x_train)
        percent_mal = []
        for i in range(n_clusters):
            idx = y_pred_train == i
            pp = (y_train[idx] == 1).sum() / idx.sum()
            percent_mal.append(pp)
        y_pred_test = kmeans.predict(x_test)
        percent_mal_test = []
        for i in range(n_clusters):
            idx = y_pred_test == i
            pp = (y_test[idx] == 1).sum() / idx.sum()
            percent_mal_test.append(pp)
        percent_mal = np.array(percent_mal)
        percent_mal_test = np.array(percent_mal_test)
        percent_mal[np.isnan(percent_mal)] = 0.
        percent_mal_test[np.isnan(percent_mal_test)] = 0.
        percent_mal_r = np.round(percent_mal, 2)
        percent_mal_test_r = np.round(percent_mal_test, 2)
        c = Counter(zip(percent_mal_r, percent_mal_test_r))
        s = [10*c[(xx,yy)] for xx,yy in zip(percent_mal_r, percent_mal_test_r)]
        plt.scatter(percent_mal_r, percent_mal_test_r, s=s)
        plt.show()

        dist = kmeans.transform(x_test)
        measurement_sequence = softmax(gamma*dist, axis=1)
        threshold = None
        error_df = None
    else:

        params_paper = {
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": 2048,
            # "num_leaves": 31,
            "max_depth": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.5
        }

        params_shallow = {
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 500,
            "learning_rate": 0.001,
            "num_leaves": 50,
            # "num_leaves": 31,
            "max_depth": 10,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.5
        }

        params_rf = {
            "boosting": "rf",
            "objective": "binary",
            "num_iterations": 500,
            "learning_rate": 0.01,
            "num_leaves": 10,
            "max_depth": 5,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.5
        }

        train_preds = []
        for i, params in zip(range(int(x_train.shape[1] / n_features) + 1), [params_paper, params_rf, params_shallow]):
            # x_train_sub = x_total[:n_train, n_features * i:n_features * (i + 1)]
            # x_test_sub = x_total[n_train:n_train+n_test, n_features * i:n_features * (i + 1)]
            x_train_sub = x_train
            x_test_sub = x_test
            if i == 1:  # always run the paper model first and then take most important features
                feat_importance = np.vstack((np.arange(2381), lgbm_model.feature_importance())).T
                feat_importance = feat_importance[feat_importance[:, 1].argsort()]
                x_train_sub = x_train[:, feat_importance[-500:, 0]]
                x_test_sub = x_test[:, feat_importance[-500:, 0]]

            train = lgb.Dataset(x_train_sub, y_train)
            file = model_file + 'model_{}'.format(i) + '.txt'
            if os.path.isfile(file):
                lgbm_model = lgb.Booster(model_file=file)
            else:
                lgbm_model = lgb.train(params, train)
                # save the model
                lgbm_model.save_model(file)

            train_preds.append(lgbm_model.predict(x_train_sub))
            measurement_sequences.append(lgbm_model.predict(x_test_sub))

        p2_act_sequence = y_test

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler.fit(x_total)
        # x_total = scaler.transform(x_total)
        # lgbm_model_all_data = lgb.train(params, lgb.Dataset(x_total, y_total))
        # feat_importance = np.vstack((np.arange(2381), lgbm_model_all_data.feature_importance())).T
        # feat_importance = feat_importance[feat_importance[:, 1].argsort()]
        # df = pd.DataFrame(x_total[:, feat_importance[-12:, 0]])
        # indep_probs = x_test[:, feat_importance[-12:, 0]]  # TODO: parameterize the number of features
        # indep_probs[indep_probs.sum(axis=1) == 0, :] = 1.
        # measurement_sequence = indep_probs / indep_probs.sum(axis=1)[:, np.newaxis]
        # outcomes = [str(i) for i in range(measurement_sequence.shape[1])]
        error_df = pd.DataFrame.from_dict({'pred_prob_{}'.format(i): measurement_sequences[i] for i in range(n_classifiers)})
        error_df['label'] = p2_act_sequence
        for i in range(n_classifiers):
            error_df['SSR_{}'.format(i)] = (error_df['pred_prob_{}'.format(i)] - error_df['label'])**2

        indep_probs = np.vstack(measurement_sequences).T
        outcomes, measurement_sequence = generate_probabilities_matrix(indep_probs)
        # measurement_sequence, outcomes = phi(indep_probs.mean(axis=1))
        # classify based on a threshold that takes into account type1 and type2 errors
        # TODO: swap grid search for analytic solution?
        mean_pred = np.vstack(train_preds).T.mean(axis=1)
        threshold = find_optimal_threshold(y_train, mean_pred, type1_weight, type2_weight)
        mean_pred_test = np.vstack(measurement_sequences).T.mean(axis=1)
        threshold_test = find_optimal_threshold(y_test, mean_pred_test, type1_weight, type2_weight)

        def threshold_and_cost(probs, tau):
            classifier_predictions = (probs > tau).astype(int)
            fp = (classifier_predictions == 1) & (p2_act_sequence == 0)
            fn = (classifier_predictions == 0) & (p2_act_sequence == 1)
            return type1_weight * fp + type2_weight * fn

        for tau in [0.25, 0.5, 0.75]:

            error_df['cost_{}'.format(tau)] = threshold_and_cost(mean_pred_test, tau)
            error_df['cost_c1_{}'.format(tau)] = threshold_and_cost(measurement_sequences[0], tau)

        test_preds = np.vstack(measurement_sequences).T.mean(axis=1)
        class_0 = test_preds[p2_act_sequence == 0]
        class_1 = test_preds[p2_act_sequence == 1]
        # Plot histograms for each class
        fig2, ax2 = plt.subplots()
        ax2.hist(class_1, bins=1000, alpha=0.5, label='Class 1', color='orange')
        ax2.hist(class_0, bins=1000, alpha=0.8, label='Class 0', color='blue')
        # Add labels and title
        ax2.set_xlabel('Classifier Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Histogram of Classifier Probabilities by Class')
        ax2.legend(loc='upper center')
        # Display the plot
        # plt.show()

    return measurement_sequence, x_test, y_test, \
        threshold, outcomes, error_df


def normalize_features(x, x_scale: dict=None):
    if x_scale is None:
        x_min = x.min(axis=0)
        x_range = x.max(axis=0) - x_min
        x_range[x_range == 0] = 1.
    else:
        x_min = x_scale['x_min']
        x_range = x_scale['x_range']

    x_norm = (x - x_min) / x_range
    x_norm[x_norm > 1.] = 1.
    x_norm[x_norm < 0.] = 0.
    return x_norm, dict(x_min=x_min, x_range=x_range)


def phi(measurement_sequence):
    thresholds = np.arange(0, 1, 0.05)
    gamma = 50.
    threshold_classes = [np.exp(-gamma * np.abs(measurement_sequence - tau)) for tau in thresholds]
    measurement_sequence = np.vstack(threshold_classes).T
    measurement_sequence = measurement_sequence / measurement_sequence.sum(axis=1)[..., np.newaxis]
    measurement_set = [str(np.round(tau, 2)) for tau in thresholds]
    return measurement_sequence, measurement_set


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn import svm, neural_network

    from LearningGames import LearningGame
    from utils import plot_simulation_results
    from benchmark_methods import MultiArmBandit, BayesianEstimator, SklearnModel
    from lstm import StreamingLSTM
    from utils import GamePlay

    finite_meas = False
    M = 600_000
    n_train = 200_000
    m_iter = int(M / 10)
    type1 = 1.
    n_classifiers = 3
    type2 = 1.
    measurement_to_label = False
    # kernel = lambda x, y: np.exp(-50 * np.linalg.norm(x - y, 2, axis=-1))
    kernel = None
    def_rng = np.random.default_rng(11)
    model_file = '../data/ember_game_{}_{}_{}_{}_{}_{}_wenergy'.format(type1, type2, n_classifiers,
        M, n_train, str(measurement_to_label))
    meas_sequence, raw_measurements, p2_act_sequence, \
        classifier_threshold, outcomes_set, classifier_perf = ember_classifier(n_train=n_train,
                                                              n_features=int(2831 / n_classifiers),
                                                              n_test=M,
                                                              model_file=model_file,
                                                              type1_weight=type1, type2_weight=type2)

    print('Measurement set: {}'.format(str(outcomes_set)))
    game = EMBERClassifierGame(opponent_action_sequence=p2_act_sequence,
                               measurement_sequence=meas_sequence, measurement_set=outcomes_set,
                               raw_measurements=raw_measurements,
                               action_set=[0, 1], type1_weight=type1, type2_weight=type2,
                               finite_measurements=finite_meas)
    dm = []
    for l, b in itertools.product([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0], [1e-2, 1e-1, 1e0]):
    # for l, b in itertools.product([1e-3], [1e-3]):
        lg = LearningGame(game.action_set, measurement_set=game.measurement_set, finite_measurements=finite_meas,
                          decay_rate=l, inverse_temperature=b, seed=0, kernel=kernel)
        lg.reset()
        dm.append(lg)

    data_window = 10_000
    update_freq = 10_000
    mab = MultiArmBandit(action_set=game.action_set, method='epsilon_greedy', method_args=dict(epsilon=0.5))
    label_action_policy = {0: 0, 1: 1}
    svm = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
                       raw_measurement=True, measurement_to_label=measurement_to_label, finite_measurement=True,
                       policy_map=label_action_policy,
                       update_frequency=update_freq, model=svm.SVR(kernel='rbf'))
    nn = neural_network.MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(30, 30))
    mlp = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
                       raw_measurement=True, measurement_to_label=measurement_to_label, finite_measurement=True,
                       policy_map=label_action_policy,
                       update_frequency=update_freq, model=nn)
    dm.append(svm)
    dm.append(mlp)

    gp = GamePlay(decision_makers=dm,
                  game=game,
                  horizon=M,
                  disp_results_per_iter=int(M/10),
                  binary_cont_measurement=False,
                  store_energy_hist=True)

    outcomes = gp.play_games()
    # gp.plot_game_outcome(outcomes, title='EMBER Streaming Dataset')

    with open(model_file + '.pkl', 'wb') as f:
        pickle.dump((outcomes, classifier_perf), file=f)

