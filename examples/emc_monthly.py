import itertools
import os

import numpy as np
import lightgbm as lgb
import pickle
import pandas as pd
import datetime as dt

from examples.simulation_utils.dataset_game import DatasetGame
from examples.simulation_utils.utils import generate_probabilities_matrix


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


def find_optimal_threshold(y_true: np.array, y_pred_proba: np.array, w_fp: float, w_fn: float):
    """
    Finds the threshold that minimizes the weighted cost w_fp * P(false pos) + w_fn * P(false neg) on a grid search
    Args:
        y_true (np.array): the true labels
        y_pred_proba (np.array): the prediction probabilities from a classifier that we want to implement a threshold for
        w_fp (float): the weight for false positive errors
        w_fn (float): the weight for false negative errors

    Returns: (float) the best threshold

    """
    thresholds = np.linspace(0, 1, 10000)
    best_threshold = thresholds[0]
    min_cost = float('inf')

    for T in thresholds:
        false_positive_rate = np.mean((y_pred_proba >= T) & (y_true == 0))
        false_negative_rate = np.mean((y_pred_proba < T) & (y_true == 1))

        total_cost = w_fp * false_positive_rate + w_fn * false_negative_rate
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = T
    print(f'Optimal threshold={best_threshold:.2f} for error weights I:{w_fp} II:{w_fn}')

    return best_threshold


def load_ember_data(ember_data_path: str):
    """
    Load the EMBER dataset
    Args:
        ember_data_path (str): folder containing the pickled ember data

    Returns: dict[str, np.ndarray]

    """
    data = dict(x_train=np.array([]), x_test=np.array([]), y_train=np.array([]), y_test=np.array([]))
    for key, _ in data.items():
        with open(ember_data_path + f'{key}.pkl', 'rb') as f:
            data[key] = pickle.load(f)
    return data


def ember_classifier(ember_data,
                     type1_weight: float, type2_weight: float, base_model_file: str, time_sort: bool) -> tuple:
    """
    Prepares the data from EMBER to be streamed in "real-time" fashion. First the data is loaded, then sorted by time-
    stamp, and finally LightGBM models are trained on n_train points to generate the stationary models. n_test points
    are stored in arrays with raw features, predictions, and ground-truth labels for the purposes of testing. The
    performance of the LightGBM models is stored in a file with the base path: base_model_file
    Args:
        type1_weight (float): the weight associated with false positives
        type2_weight (float): the weight associated with false negatives
        base_model_file (str): the file path for the models to be persisted to

    Returns: (tuple): measurement_sequence (np.ndarray), x_test (np.ndarray), y_test (np.ndarray), \
        threshold (float), outcomes (list[str]), error_df (pd.DataFrame), indep_probs (np.ndarray)

    """

    x_total = np.vstack((ember_data['x_train'], ember_data['x_test']))
    y_total = np.hstack((ember_data['y_train'], ember_data['y_test'])).T
    # load metadata about dates
    meta_df = pd.read_json('../data/extracted_data.jsonl', lines=True)
    # only include data in 2018 to prevent training only on benign samples, which is all of them pre-2018
    # if not time_sort:
    #     cutoff_idx = np.where(pd.to_datetime(meta_df['appeared']) >= dt.datetime(2018, 1, 1))[0][0]
    #     x_total = x_total[cutoff_idx:]
    #     y_total = y_total[cutoff_idx:]
    dates = pd.to_datetime(pd.Series(meta_df['appeared']))
    time_array = dates.dt.month
    # only include the labeled points
    x_total = x_total[y_total >= 0.]
    time_array = time_array[y_total >= 0.]
    y_total = y_total[y_total >= 0.]
    n_train = np.where(dates <= dt.datetime(2018, 1, 15))[-1][-1]
    # test on the first month so that models will have access to data for second month
    n_test = np.where(dates >= dt.datetime(2018, 1, 1))[0][0]
    n_points = len(y_total)
    print(f'{n_points} points')
    # break into train and test sets
    x_test = x_total[n_test:]
    # scale the features (not strictly necessary)
    # x_train, x_scale = normalize_features(x_train)
    # x_test, _ = normalize_features(x_test, x_scale)

    measurement_sequences = []
    y_test = y_total[n_test:]
    time_array = time_array.values  # TODO: clean up code
    time_array = time_array[n_test:] - time_array[n_test]  # make the time start at zero

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

    train_preds = []
    n_train_start = [50_000, 0, 0]
    n_train_list = [n_train, 100_000, n_train]
    for i, params in zip(range(n_classifiers), [params_paper, params_paper, params_paper]):
        x_train_sub = x_total[n_train_start[i]:n_train_list[i]]
        x_test_sub = x_test
        y_train_sub = y_total[n_train_start[i]:n_train_list[i]]

        train = lgb.Dataset(x_train_sub, y_train_sub)
        file = base_model_file + '_model_{}'.format(i) + '.txt'
        if os.path.isfile(file):
            lgbm_model = lgb.Booster(model_file=file)
        else:
            lgbm_model = lgb.train(params, train)
            # save the model
            lgbm_model.save_model(file)

        train_preds.append(lgbm_model.predict(x_train_sub))
        measurement_sequences.append(lgbm_model.predict(x_test_sub))

    p2_act_sequence = y_test

    error_df = pd.DataFrame.from_dict(
        {'pred_prob_{}'.format(i): measurement_sequences[i] for i in range(n_classifiers)})

    indep_probs = np.vstack(measurement_sequences).T
    # outcomes, measurement_sequence = generate_probabilities_matrix(indep_probs)
    measurement_sequence, outcomes = phi_grid(indep_probs)
    # classify based on a threshold that takes into account type1 and type2 errors
    # mean_pred = np.vstack(train_preds).T.mean(axis=1)
    # threshold = find_optimal_threshold(y_train, mean_pred, type1_weight, type2_weight)
    mean_pred_test = np.vstack(measurement_sequences).T.mean(axis=1)
    threshold_test = find_optimal_threshold(y_test, mean_pred_test, type1_weight, type2_weight)

    def threshold_and_cost(probs, t):
        classifier_predictions = (probs > t).astype(int)
        fp = (classifier_predictions == 1) & (p2_act_sequence == 0)
        fn = (classifier_predictions == 0) & (p2_act_sequence == 1)
        return type1_weight * fp + type2_weight * fn

    for tau in [0.25, 0.5, 0.75]:
        error_df['cost_{}'.format(tau)] = threshold_and_cost(mean_pred_test, tau)
        error_df['cost_c1_{}'.format(tau)] = threshold_and_cost(measurement_sequences[0], tau)

    error_df['label'] = p2_act_sequence
    error_df['appeared'] = pd.to_datetime([f'2018-{m+1:02d}-01' for m in time_array])
    for i in range(n_classifiers):
        error_df['mae_{}'.format(i)] = np.abs(error_df['pred_prob_{}'.format(i)] - error_df['label'])

    with open(base_model_file + f'{type1_weight}_{type2_weight}_classifier.pkl', 'wb') as f:
        pickle.dump(error_df, f)

    return measurement_sequence, x_test, y_test, \
        threshold_test, outcomes, error_df, indep_probs, time_array


def phi_grid(measurement_sequence):
    thresholds = np.arange(0, 1, .2)
    gamma = 50.
    measurement_set = [str(np.round(t1, 2))+str(np.round(t2, 2))+str(np.round(t3, 2))
                       for t1, t2, t3 in itertools.product(thresholds, thresholds, thresholds)]
    threshold_classes = [np.exp(-gamma * np.linalg.norm(measurement_sequence - np.array([t1, t2, t3]), axis=1))
                         for t1, t2, t3 in itertools.product(thresholds, thresholds, thresholds)]
    measurement_sequence_out = np.vstack(threshold_classes).T
    measurement_sequence_out = measurement_sequence_out / measurement_sequence_out.sum(axis=1)[..., np.newaxis]
    return measurement_sequence_out, measurement_set


def vectorized_ml_play(x_raw, time_array, game, y, svm, mlp):
    cost_over_months = dict()
    action_0 = np.zeros((len(x_raw), 2))
    action_0[:, 0] = 1.
    action_1 = np.zeros((len(x_raw), 2))
    action_1[:, 0] = 1.
    for model in [svm.model, mlp.model]:
        key = model.__class__.__name__
        for month in np.unique(time_array)[1:]:
            print(month)
            prev_idx = time_array == month-1
            current_idx = time_array == month
            y_train_sub = y[prev_idx]
            y_test_sub = y[current_idx]
            pred_costs = []
            for act_i, act in enumerate([action_0, action_1]):
                costs = np.array([game.cost(act_i, y) for y in y_train_sub])
                # x_total = np.hstack((raw_measurements, act))
                x_train = x_raw[prev_idx]
                # y_train = p2_act_sequence[prev_idx]
                x_test = x_raw[current_idx]
                # y_test = p2_act_sequence[current_idx]
                model.fit(x_train, costs)
                pred_costs.append(model.predict(x_test))
            pred_cost_arr = np.array(pred_costs).T
            sel_action = np.argmin(pred_cost_arr, axis=1)
            eval_cost = np.array([game.cost(my_act, y) for my_act, y in zip(sel_action, y_test_sub)])
            cost_over_months[key + f'_{month}'] = eval_cost
    with open(f'ember_monthly_{type1}_{type2}_benchmark_costs.pkl', 'wb') as f:
        pickle.dump(cost_over_months, file=f)
    print('Finished ML games')


def normalize_features(x: np.ndarray, x_scale: dict = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Scale the feature space to be [0,1] for the training set and use the same scale factors for the test set
    Args:
        x (np.ndarray): the data to scale
        x_scale (dict, optional): the min values and range

    Returns: (np.array, dict[str, np.ndarray])

    """
    if x_scale is None:
        x_min = x.min(axis=0)
        x_range = x.max(axis=0) - x_min
        x_range[x_range == 0] = 1.
    else:
        x_min = x_scale['x_min']
        x_range = x_scale['x_range']

    x_norm = (x - x_min) / x_range
    return x_norm, dict(x_min=x_min, x_range=x_range)


def name_file(use_raw_measurement, measurement_to_label, store_energy, unit_step):
    """Helper func to name files uniquely based on test parameters"""
    if unit_step:
        step = '_unit'
    else:
        step = ''
    base_model_file = f'../data/ember_monthly'
    raw_str = 'nonraw'
    if use_raw_measurement:
        raw_str = 'raw'
    mapping_str = 'ya2j'
    if measurement_to_label:
        mapping_str = 'y2l'
    energy_str = 'noenergy'
    if store_energy:
        energy_str = 'wenergy'
    total_file = base_model_file + f'{step}_{type1}_{type2}_{mapping_str}_{raw_str}_{energy_str}'
    if not os.path.isdir(total_file):
        os.mkdir(total_file)
    return base_model_file, total_file


if __name__ == '__main__':

    from sklearn import svm, neural_network

    from learning_games import LearningGame
    from examples.benchmark_methods import SklearnModel
    from examples.simulation_utils.utils import GamePlay

    ember_filepath: str = '../ember_data/'  # wherever you stored the ember dataset
    # (assumes files are x_train.pkl, y_train.pkl, x_test.pkl, y_test.pkl)
    sort_by_header_timestamp = False
    m_iter: int = 20_000  # the printout frequency from game play
    type1: float = 1.  # the weight for false pos
    type2: float = 1.  # the weight for false neg
    store_energy = False  # True: stores the energy and persists with rest of simulation data but is memory intensive

    # specify beta and lambda values to search over. We later compute the cartesian product of these lists/sets.
    lambda_values: list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]
    beta_values: list = [1e-2, 1e-1, 1e0]
    lambda_values: list = [1e-3, 1e-4, 1e-5]
    beta_values: list = [1e4, 1e5, 1e6]

    # parameters for the benchmark models
    use_raw_measurement: bool = True  # whether to use raw features (True) or the probabilistic classifier as input
    # whether to learn a map from measurements to label (True) or estimate the cost associated with each action (False)
    measurement_to_label: bool = False
    # whether to use a unit step or the actual timestamps from the data
    unit_step: bool = True
    # the window size H of recent data
    data_window: int = 500_000
    # the frequency at which the model is updated. More updates requires more time
    update_freq: int = 1_000_000 # make arbitarily large so it only will update with new month
    # the mapping from labels to actions when learning measurement_to_label
    label_action_policy = {0: 0, 1: 1}
    # hidden layer sizes for NN/MLP
    hidden_layer_sizes = (50, 50)
    # the max number of training iterations for the NN
    max_train_iter = 1000
    # kernel choice for the SVM
    svm_kernel = 'rbf'
    # random state for the sklearn models
    random_state = 1

    # Not easy to change parameters. Likely require some downstream changes
    n_classifiers: int = 3  # currently can't be changed without modifying how the classifiers are generated
    finite_meas: bool = False  # whether to use finite or infinite measurements

    # file naming convention
    base_model_file, total_file_dir = name_file(measurement_to_label=measurement_to_label,
                                            use_raw_measurement=use_raw_measurement,
                                            store_energy=store_energy, unit_step=unit_step)
    print(total_file_dir)
    # load ember dataset
    dataset = load_ember_data(ember_filepath)

    # preprocess ember dataset into useful objects
    meas_sequence, raw_measurements, p2_act_sequence, \
        classifier_threshold, outcomes_set, classifier_perf, classifier_probs, time_diff = ember_classifier(ember_data=dataset,
                                                                               time_sort=sort_by_header_timestamp,
                                                                               base_model_file=base_model_file,
                                                                               type1_weight=type1, type2_weight=type2)
    M = len(time_diff)
    # set raw measurements to classifier probs # TODO: formalize this so there are 3 options for ML input
    raw_measurements = classifier_probs

    print('Measurement set: {}'.format(str(outcomes_set)))
    # Create the dataset game
    game = EMBERClassifierGame(opponent_action_sequence=p2_act_sequence,
                               measurement_sequence=meas_sequence, measurement_set=outcomes_set,
                               raw_measurements=raw_measurements,
                               action_set=[0, 1], type1_weight=type1, type2_weight=type2,
                               finite_measurements=finite_meas)
    dm = []
    for lam, beta in itertools.product(lambda_values, beta_values):
        # create a number of different Boltzmann learning objects with different hyperparameters
        lg = LearningGame(game.action_set, measurement_set=game.measurement_set, finite_measurements=finite_meas,
                          decay_rate=lam, inverse_temperature=beta, seed=0, compute_entropy=False)
        lg.reset()
        dm.append(lg)
    # create an SVM and MLP as alternatives to Boltzmann learning
    if measurement_to_label:
        nn_model = neural_network.MLPClassifier(random_state=random_state,
                                                max_iter=max_train_iter,
                                                hidden_layer_sizes=hidden_layer_sizes)
        svm_model = svm.SVC(kernel=svm_kernel)
    else:
        nn_model = neural_network.MLPRegressor(random_state=random_state,
                                               max_iter=max_train_iter,
                                               hidden_layer_sizes=hidden_layer_sizes)
        svm_model = svm.SVR(kernel=svm_kernel)
    svm = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
                       raw_measurement=use_raw_measurement, measurement_to_label=measurement_to_label,
                       finite_measurement=finite_meas, policy_map=label_action_policy,
                       update_frequency=update_freq, model=svm_model,
                       use_time_index=True)

    mlp = SklearnModel(window_size=data_window, action_set=game.action_set, measurement_set=game.measurement_set,
                       raw_measurement=use_raw_measurement, measurement_to_label=measurement_to_label,
                       finite_measurement=finite_meas, policy_map=label_action_policy,
                       update_frequency=update_freq, model=nn_model,
                       use_time_index=True)
    # play the games in a vectorized fashion for speed, otherwise it will take much (>>) longer
    vectorized_ml_play(x_raw=raw_measurements,
                       time_array=time_diff,
                       game=game,
                       y=p2_act_sequence,
                       svm=svm,
                       mlp=mlp)

    # dm = []
    # dm.append(svm)
    # dm.append(mlp)
    dm.reverse()

    if unit_step:
        time_diff = None

    # Construct the harness to play the games
    gp = GamePlay(decision_makers=dm,
                  game=game,
                  horizon=M,
                  disp_results_per_iter=m_iter,
                  store_energy_hist=store_energy,
                  time_index=time_diff)
    # play all the games
    gp.play_games(save_to=total_file_dir)
