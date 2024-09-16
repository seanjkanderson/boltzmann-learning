from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from typing import Union


class DatasetGame(metaclass=ABCMeta):

    def __init__(self, action_set: list, measurement_set: list, opponent_action_sequence,
                 measurement_sequence, finite_measurements: bool):
        """Create game

        Args:
            action_set (list): the set of possible actions
            measurement_set (list): the set of possible measurements in the finite measurement case and the set of
                classes for the infinite measurement case
            opponent_action_sequence (Array-like): sequence of actions for player 2
            measurement_sequence (Array-like): sequence of measurements. For the finite measurement case, this will be a
                vector. For the infinite measurement case, this will be an array with columns matching the order of
                measurement_set

        """
        self.opponent_action_sequence = opponent_action_sequence.squeeze()
        self.measurement_sequence = measurement_sequence.squeeze()
        self.counter = 0
        self.action_set = action_set
        self.measurement_set = measurement_set
        self.finite_measurements = finite_measurements
        # initialize measurement
        if self.finite_measurements:
            self.measurement = measurement_set[0]
        else:
            self.measurement = {k: 0. for k in measurement_set}
            self.measurement[measurement_set[0]] = 1.  # the measurements must sum to 1.

    @abstractmethod
    def cost(self, p1_action, p2_action) -> float:
        """Returns game outcome

        Args:
            p1_action: action for player 1
            p2_action: action for player 2

        Returns:
            game result (float)
        """

    def get_measurement(self):
        """get measurement for next move

        Returns:
            measurement: measurement for next move
        """
        return self.measurement

    def play(self, p1_action) -> tuple[float, OrderedDict[int, float], Union[str, float]]:
        """Play game

        Args:
            p1_action: action for player 1

        Returns:
            tuple[float,dict(str,float)]
            cost (float): -1 if player 1 wins, +1 if player 1 loses, 0 draw
            all_costs (dict[str,float]): dictionary with costs for all actions of player 1
            p2_action
        """
        if self.counter >= len(self.opponent_action_sequence) or self.counter + 1 >= len(self.measurement_sequence):
            raise ValueError('player 2 ran out of actions or there are no more measurements')

        # select action for player 2 (i.e. read topic for label/cost/opponent action.)
        p2_action = self.opponent_action_sequence[self.counter]
        # update measurement (i.e. read topic for measurements)
        self.counter += 1
        all_measurements = self.measurement_sequence[self.counter]
        if self.finite_measurements:
            self.measurement = all_measurements
        else:
            self.measurement = {key: meas for key, meas in zip(self.measurement_set, all_measurements)}
        # cost for given action of player 1
        cost = self.cost(p1_action, p2_action)
        # costs for all actions of player 1
        all_costs = OrderedDict([(a, self.cost(a, p2_action)) for a in self.action_set])

        return cost, all_costs, p2_action
