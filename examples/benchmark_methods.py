from copy import deepcopy
from typing import Any

from sklearn.base import BaseEstimator
import numpy as np

from decision_maker import DecisionMaker


class SklearnModel(DecisionMaker):
    """Class to implement sklearn models as decision makers."""
    def __init__(self, model: BaseEstimator, window_size: int, measurement_set: list, action_set: list,
                 finite_measurement: bool,
                 update_frequency: int = 1, raw_measurement: bool = False, measurement_to_label: bool = False,
                 policy_map: dict = None, use_time_index: bool = False):
        """

        Args:
            model (BaseEstimator): an estimator model from sklearn
            window_size (int): the length of the hard window of past data the model will access
            measurement_set (list): the list (set-like) of measurements
            action_set (list): the list (set-like) of actions
            finite_measurement (bool): True for finite measurement setting and False for continuum setting
            update_frequency (int, optional): the frequency at which the model should be (re)trained
                Default is 1 (i.e. every iteration), but this is very slow due to model retraining
            raw_measurement (bool, optional): whether to use the raw measurements, typically corresponding to a
                high-dimensional feature space (True) or use lower-dimensional features
                    (e.g. output from prob. classifier)
                Default to False (i.e. use processed measurements)
            measurement_to_label (bool, optional):
            policy_map:
            use_time_index: TODO: finish
        """

        self.model_copy = deepcopy(model)  # ensures there is no data leakage
        self.measurement_set = measurement_set
        self.finite_measurement = finite_measurement
        self.update_frequency = update_frequency
        self.raw_measurement = raw_measurement
        self.measurement_to_label = measurement_to_label
        self.policy_map = policy_map
        self.window_size = window_size * len(action_set)
        if self.measurement_to_label:
            self.window_size = window_size
            if self.policy_map is None:
                raise ValueError('Mapping from measurement to label requires a policy '
                                 'map/dict to map labels to actions.')
        self.inputs = []
        self.outputs = []
        self.time_index = []
        self.previous_time = 0

        # map the measurement set to a one-hot encoding
        self.model = model
        self.model_degenerate = True
        self.encoding_map = self.one_hot_encoding(measurement_set)
        self.action_set = action_set
        self.action_encoding_map = self.one_hot_encoding(action_set)
        self.use_time_index = use_time_index

    @staticmethod
    def one_hot_encoding(option_set: list) -> dict:
        """
        Maps the set of options to one hot encodings
        Args:
            option_set (list): the set of possible outcomes

        Returns: (dict) the mapping from a measurement to the corresponding one hot encoding

        """
        n_elements = len(option_set)
        one_hot = np.eye(n_elements)
        encoding_map = dict()
        for idx, meas in enumerate(option_set):
            encoding_map[meas] = one_hot[idx]
        return encoding_map

    def measurement_to_embedding(self, measurement: dict, raw_measurement: Any=None, action: Any=None) -> np.array:
        """
        Defines the embedding of measurements for all cases.
        Args:
            measurement (dict): the current measurement (keys are measurement classes in measurement_set)
            raw_measurement (Any, optional): raw measurement correponding to higher dimensional space
            action (Any, optional): current action corresponding to the measurements

        Returns: (np.array) embedding of the measurement corresponding to the feature space

        """
        if self.raw_measurement:
            embedding = raw_measurement
        else:
            if not self.finite_measurement:
                embedding = np.array([measurement[k] for k in self.measurement_set])
            else:
                embedding = self.encoding_map[measurement]

        if self.measurement_to_label:
            return embedding
        else:
            one_hot_action = self.action_encoding_map[action]
            return np.hstack((embedding, one_hot_action))

    def get_action(self, measurement: dict, time: float = 0.0, **kwargs):
        """
        Equivalent to a BaseEstimator.predict method but goes a step further to select an action
        Args:
            measurement (dict): the current measurement, which will form (part of) the feature space
            time (float): the current time
            **kwargs: raw_measurement (optional), contains the unprocessed features or is None

        Returns: (action (Any), None, None) with second and third arguments as placeholders to satisfy comparison with
            other decision makers

        """
        if self.model_degenerate:
            return 0, None, None
        else:
            min_cost = 1e10
            best_action = ''
            if self.measurement_to_label:
                embedding = self.measurement_to_embedding(measurement=measurement,
                                                          raw_measurement=kwargs['raw_measurement'])
                input = np.hstack(embedding).reshape(1, -1)
                predicted_label = self.model.predict(input)[0]
                best_action = self.policy_map[predicted_label]
            else:
                for action in self.action_set:
                    embedding = self.measurement_to_embedding(measurement=measurement,
                                                              raw_measurement=kwargs['raw_measurement'],
                                                              action=action)
                    input = np.hstack(embedding).reshape(1, -1)
                    predicted_cost = self.model.predict(input)[0]
                    if predicted_cost < min_cost:
                        min_cost = predicted_cost
                        best_action = action
            return best_action, None, None

    def update_energies(self, measurement: dict, costs: dict, time: float = 0.0, **kwargs):
        """
        Update the models. Generically this is a "fit" stage.
        Args:
            measurement (dict): the current measurement, which will form (part of) the feature space
            costs (dict): keys are the actions (in action_set) and values are the associated costs
            time (float): the current time
            **kwargs: raw_measurement (optional), contains the unprocessed features or is None
                    opponent_action (optional), contains the opponent's action
        """
        if self.measurement_to_label:
            embedding = self.measurement_to_embedding(measurement, raw_measurement=kwargs['raw_measurement'])
            self.inputs.append(embedding)
            self.outputs.append(kwargs['opponent_action'])
        else:
            for action, cost in costs.items():
                embedding = self.measurement_to_embedding(measurement=measurement,
                                                          raw_measurement=kwargs['raw_measurement'], action=action)
                self.inputs.append(embedding)
                self.outputs.append(cost)

        self.inputs = self.inputs[-self.window_size:]
        self.outputs = self.outputs[-self.window_size:]

        if time % self.update_frequency == 0:
            try:
                self.model = deepcopy(self.model_copy)
                self.model.fit(np.array(self.inputs), np.array(self.outputs))
                self.model_degenerate = False
                print(f'(Re)trained model. {len(self.inputs)} samples')
            except ValueError as e:
                print('model is degenerate with {} samples | {}'.format(len(self.outputs), e))
                self.model_degenerate = True


class MultiArmBandit(DecisionMaker):

    def __init__(self, action_set, method: str, method_args: dict):
        self.action_set = action_set
        self.n_arms = len(action_set)
        self.q = np.zeros(self.n_arms)  # init average cost
        self.count = np.zeros(self.n_arms)  # init arm-pull counter
        self.sum_rewards = np.zeros(self.n_arms)  # init average cost

        if method == 'epsilon_greedy':  # TODO: should be able to pass string in as direct reference to method names
            self.arm_selector = self.epsilon_greedy
        self.method_args = method_args

    def get_action(self,  measurement, time: float = 0.0, **kwargs):
        # TODO: should be able to compute regret from distribution of actions
        action_idx = self.arm_selector(t=time, measurement=measurement)
        return self.action_set[action_idx], None, None

    def update_energies(self, action, action_cost, **kwargs):
        action_idx = self.action_set.index(action)
        self.count[action_idx] += 1
        self.sum_rewards[action_idx] += action_cost
        self.q[action_idx] = self.sum_rewards[action_idx]/self.count[action_idx]

    def epsilon_greedy(self, **kwargs):

        rand = np.random.random()
        if rand < self.method_args['epsilon']:
            action = np.random.choice(self.n_arms)
        else:
            action = np.argmin(self.q)  # typically argmax for reward maximization

        return action

    def softmax(self, **kwargs):

        total = sum([np.exp(val / self.method_args['tau']) for val in self.q])
        probs = [np.exp(val / self.method_args['tau']) / total for val in self.q]

        threshold = np.random.random()
        cumulative_prob = 0.0
        for i in range(len(probs)):
            cumulative_prob += probs[i]
            if (cumulative_prob > threshold):
                return i
        return np.argmin(probs)

    def thompson_sampling(self, **kwargs):

        samples = [np.random.beta(self.method_args['alpha'][i] + 1,
                                  self.method_args['beta'][i] + 1) for i in range(10)]

        return np.argmax(samples)


class BayesianEstimator(DecisionMaker):
    """Class that implements a Bayesian estimator that assumes stationarity."""
    def __init__(self, action_set: list, measurement_set: list):
        """

        Args:
            action_set (list): the set of possible actions
            measurement_set (list): the set of possible measurements
        """
        self.action_set = action_set
        self.measurement_set = measurement_set
        self.n_arms = len(action_set)
        self.q = np.zeros((self.n_arms, len(measurement_set)))  # init average cost

    def get_action(self, measurement, time: float = 0.0, **kwargs):
        """
        Equivalent to a generic predict method but goes a step further to select an action
        Args:
            measurement (dict): the current measurement, which will form (part of) the feature space
            time (float): the current time

        Returns: (action (Any), None, None) with second and third arguments as placeholders to satisfy comparison with
            other decision makers

        """
        measurement_idx = self.measurement_set.index(measurement)
        action_idx = np.argmin(self.q[:, measurement_idx], axis=0)
        return self.action_set[action_idx], None, None

    def update_energies(self, measurement, costs, **kwargs):
        """
        Update the models. Generically this is a "fit" stage.
        Args:
            measurement (dict): the current measurement, which will form (part of) the feature space
            costs (dict): keys are the actions (in action_set) and values are the associated costs
            time (float): the current time
            **kwargs: raw_measurement (optional), contains the unprocessed features or is None"""
        measurement_idx = self.measurement_set.index(measurement)
        cost_arr = np.array([costs[key] for key in self.action_set])  # assumes action set is ordered
        self.q[:, measurement_idx] += cost_arr
