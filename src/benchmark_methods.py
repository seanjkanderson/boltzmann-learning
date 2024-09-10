from sklearn import svm
import numpy as np

from decision_maker import DecisionMaker


class SVMRegressor(DecisionMaker):

    def __init__(self, window_size: int, kernel: str, measurement_set, action_set):
        self.window_size = window_size * len(action_set)
        self.inputs = []
        self.outputs = []

        # map the measurement set to a one-hot encoding
        self.model = svm.SVR(kernel=kernel)
        self.model_degenerate = True
        self.encoding_map = self.one_hot_encoding(measurement_set)
        self.action_set = action_set
        self.action_encoding_map = self.one_hot_encoding(action_set)

    @staticmethod
    def one_hot_encoding(measurement_set):
        n_elements = len(measurement_set)
        one_hot = np.eye(n_elements)
        encoding_map = dict()
        for idx, meas in enumerate(measurement_set):
            encoding_map[meas] = one_hot[idx]
        return encoding_map

    def get_action(self, measurement, time: float = 0.0, **kwargs):
        one_hot_measurement = self.encoding_map[measurement]
        if self.model_degenerate:
            return 0, None, None
        else:
            min_cost = 1e10
            best_action = ''
            for action in self.action_set:
                one_hot_action = self.action_encoding_map[action]
                input = np.hstack((one_hot_measurement, one_hot_action)).reshape(1, -1)
                predicted_cost = self.model.predict(input)[0]
                if predicted_cost < min_cost:
                    min_cost = predicted_cost
                    best_action = action
            return best_action, None, None

    def update_energies(self, measurement, costs: dict, time: float = 0.0, **kwargs):
        for action, cost in costs.items():
            one_hot_action = self.action_encoding_map[action]
            self.inputs.append(np.hstack((self.encoding_map[measurement], one_hot_action)))
            self.outputs.append(cost)
        self.inputs = self.inputs[-self.window_size:]
        self.outputs = self.outputs[-self.window_size:]
        try:
            self.model.fit(np.array(self.inputs), np.array(self.outputs))
            self.model_degenerate = False
        except:
            print('model is degenerate with {} samples'.format(len(self.outputs)))
            self.model_degenerate = True


class MultiArmBandit:

    def __init__(self, action_set, method: str, method_args: dict):
        self.action_set = action_set
        self.n_arms = len(action_set)
        self.q = np.zeros(self.n_arms)  # init average cost
        self.count = np.zeros(self.n_arms)  # init arm-pull counter
        self.sum_rewards = np.zeros(self.n_arms)  # init average cost

        if method == 'epsilon_greedy':  # TODO: should be able to pass string in as direct reference to method names
            self.arm_selector = self.epsilon_greedy
        self.method_args = method_args

    def get_action(self, t, measurement):
        # TODO: should be able to compute regret from distribution of actions
        action_idx = self.arm_selector(t=t, measurement=measurement)
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

    # def ucb(self, **kwargs):
    #
    #     ucb = np.zeros(10)
    #
    #     # explore all the arms
    #     if kwargs['iters'] < 10:
    #         return i
    #
    #     else:
    #         for arm in range(10):
    #             # calculate upper bound
    #             upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])
    #
    #             # add upper bound to the Q valyue
    #             ucb[arm] = Q[arm] + upper_bound
    #
    #         # return the arm which has maximum value
    #         return (np.argmax(ucb))


class MultiArmBanditFullInfo(DecisionMaker):

    def __init__(self, action_set: list, measurement_set: list):
        self.action_set = action_set
        self.measurement_set = measurement_set
        self.n_arms = len(action_set)
        self.q = np.zeros((self.n_arms, len(measurement_set)))  # init average cost

    def get_action(self, measurement, time: float = 0.0):
        measurement_idx = self.measurement_set.index(measurement)
        action_idx = np.argmin(self.q[:, measurement_idx], axis=0)
        return self.action_set[action_idx], None, None

    def update_energies(self, measurement, costs, **kwargs):
        measurement_idx = self.measurement_set.index(measurement)
        cost_arr = np.array([costs[key] for key in self.action_set])  # assumes action set is ordered
        self.q[:, measurement_idx] += cost_arr
