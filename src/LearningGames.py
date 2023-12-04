#!../venv/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 17:57 2024

@author: Joao Hespanha
"""

import numpy

import typing

Action = typing.NewType("Action", typing.Any)
Measurement = typing.NewType("Measurement", typing.Any)

DEBUG = False


def get_random_integer(rng: numpy.random, p: numpy.array) -> int:
    """generate random integer for a given probability distribution

    Args:
        rng (numpy.random): random number generator
        p (numpy.array): vector with probabilities, which must add up to 1.0

    Returns:
        int: random integer from 0 to len(p)-1
    """
    cum = numpy.cumsum(p)
    uniform_rand = rng.random()
    k = numpy.sum(uniform_rand > cum)
    return k


class LearningGame:
    """Class to learn no-regret policy"""

    def __init__(
        self,
        action_set: set[Action],
        measurement_set: set[Measurement] = {None},
        gamma: float = 0.0,
        inverse_temperature: float = 0.01,
        seed: int = None,
    ):
        """Constructor for LearningGame class

        Args:
            action_set (set[Action]): set of all possible actions
            measurement_set (set[Measurement], optional): set of all possible measurements.
                Defaults to {None}.
            gamma (float, optional): information decay rate (0.0 means no decay).
                Defaults to 0.0.
            inverse_temperature (float, optional): inverse of thermodynamic temperature
                for the Boltzmann distribution.
                Defaults to 10.
            seed (int, optional): seed for action random number generator.
                When None, a nondeterministic seed will be generated.
                Defaults to None.
        """

        ## parameters
        self.action_set = frozenset(action_set)  # to make it immutable
        """action_set (set[Action]): set of all possible actions"""

        self.measurement_set = frozenset(measurement_set)  # to make it immutable
        """measurement_set (set[Measurement]): set of all possible measurements"""

        self.gamma = gamma
        """gamma (float): information decay rate (0.0 means no decay)"""

        self.inverse_temperature = inverse_temperature
        """inverse_temperature (float): inverse of thermodynamic temperature for the Boltzmann distribution."""

        ## initialize random number generator
        self.rng = numpy.random.default_rng(seed)
        """rng (numpy.random._generator.Generator): random number generator to select actions"""

        self.reset()

    def reset(self):
        """Reset all parameters: total cost, min/max costs, energies"""
        ## initialize variables to compute regret
        self.total_cost = 0.0
        """average_cost (float): average cost incurred so far; used to compute regret"""

        self.number_updates = 0
        """number_updates (int): total number of updates"""

        ## initialize bounds
        self.min_cost = 0  # The result requires min_cost<=0
        """min_cost (float): minimum value of the cost"""
        self.max_cost = 0  # The result requires max_cost>=0
        """max_cost (float): maximum value of the cost"""

        ## initialize energies
        self.energy = {
            y: {a: 0.0 for a in self.action_set} for y in self.measurement_set
        }
        """energy (dict[Measurement:dict[Action:float]] 
            self.energy[y][a] (float): energy associated with action a and measurements y
        """

    def get_action(self, measurement: Measurement):
        """Gets (optimal) action for a given measurement

        Args:
            measurement (Measurement): measurement, which must be an element of self.measurement_set

        Returns:
            action (Action): (optimal) action
        """
        probabilities = self.get_Boltzmann_distribution(measurement)
        # get integer
        action_index = get_random_integer(self.rng, probabilities)
        # convert to action
        action = list(self.action_set)[action_index]
        return action

    def get_Boltzmann_distribution(self, measurement: Measurement):
        """Returns a Boltzmann distribution

        Args:
            measurement (Measurement): measurement, which must be an element of self.measurement_set

        Returns:
            probabilities (numpy.array)
        """
        # TODO: It seems a little dangerous to rely on set order to construct the probability array
        #        An alternative would be to keep probabilities as a dict, but this would be less efficient

        # compute Boltzmann distribution
        energies_array = numpy.array([v for (k, v) in self.energy[measurement].items()])
        min_energy = numpy.min(energies_array)
        probabilities = numpy.exp(
            -self.inverse_temperature * (energies_array - min_energy)
        )
        probabilities = probabilities / numpy.sum(probabilities)
        return probabilities

    def update_energies(self, measurement: Measurement, costs: dict[Action, float]):
        """Updates energies based on after-the-fact costs

        Args:
            measurement (Measurement): measurement, which must be an element of self.measurement_set
            costs (dict[Action,float]): costs for each action
        """

        ## update bounds
        for k, a in enumerate(self.action_set):
            if costs[a] < self.min_cost:
                self.min_cost = costs[a]
            if costs[a] > self.max_cost:
                self.max_cost = costs[a]

        ## update regrets
        average_cost = 0
        probabilities = self.get_Boltzmann_distribution(measurement)
        for k, a in enumerate(self.action_set):
            average_cost += probabilities[k] * costs[a]
        self.total_cost = (1 - self.gamma) * self.total_cost + average_cost

        self.number_updates += 1

        ## update all energies
        for m in self.measurement_set:
            if m == measurement:
                for a in self.action_set:
                    self.energy[m][a] = (1 - self.gamma) * self.energy[m][a] + costs[a]
            else:
                for a in self.action_set:
                    self.energy[m][a] = (1 - self.gamma) * self.energy[m][a]

    def get_regret(
        self, display=False
    ) -> tuple[float, float, float, float, float, int, float, float]:
        """Computes regret based on after-the-fact costs in update_energies()

        Returns:
            tuple[float, float, float, float, float, int]:
                average_cost (float): average cost incurred
                minimum_cost (float): minimum cost for stationary policy
                cost_bound (float): theoretical bound on cost
                regret (float): cost regret
                regret_bound (float): theoretical bound on regret
                        regret_bound = alpha1 * minimum_cost +alpha0
                number_updates (int): number of update steps
                alpha1 (float): multiplicative factor for theoretical bound
                alpha0 (float): additive factor for theoretical bound

        """

        ## compute average cost
        if self.gamma == 0.0:
            if self.number_updates > 0:
                average_cost = self.total_cost / self.number_updates
            else:
                average_cost = 0.0
        else:
            average_cost = self.gamma * self.total_cost

        ## compute average minimum cost
        minimum_cost = 0
        for m in self.measurement_set:
            mn = numpy.inf
            for a in self.action_set:
                if self.energy[m][a] < mn:
                    mn = self.energy[m][a]
            minimum_cost += mn

        if self.gamma == 0.0:
            if self.number_updates > 0:
                minimum_cost /= self.number_updates
        else:
            minimum_cost *= self.gamma

        regret = average_cost - minimum_cost

        ## compute bounds
        J0 = self.min_cost  # TODO: there may be a better choice
        J0 = (self.min_cost + self.max_cost) / 2  # TODO: there may be a better choice
        J0 = minimum_cost / (1 - self.gamma) / len(
            self.measurement_set
        ) + self.gamma / (1 - self.gamma) * (
            numpy.log(len(self.action_set)) / self.inverse_temperature - self.min_cost
        )  # TODO: there may be a better choice
        delta = (
            numpy.exp(self.inverse_temperature * (J0 - self.min_cost))
            - numpy.exp(self.inverse_temperature * (J0 - self.max_cost))
        ) / (self.inverse_temperature * (self.max_cost - self.min_cost))
        delta0 = (
            1
            - numpy.exp(self.inverse_temperature * (J0 - self.min_cost))
            + self.inverse_temperature * (J0 - delta * self.min_cost)
        )
        alpha1 = 1 / (1 - self.gamma) / delta
        alpha00 = (
            -delta0 * (1 - self.gamma)
            - self.inverse_temperature * self.gamma * self.min_cost
        )

        if self.gamma == 0.0:
            # print("   gamma == 0")
            alpha0 = (
                len(self.measurement_set)
                * numpy.log(len(self.action_set))
                / (self.inverse_temperature * self.number_updates)
                + alpha00 * len(self.measurement_set) / self.inverse_temperature
            )
        else:
            # print("   gamma != 0")
            alpha0 = self.gamma * len(self.measurement_set) * numpy.log(
                len(self.action_set)
            ) / self.inverse_temperature + alpha00 * len(
                self.measurement_set
            ) / self.inverse_temperature * (
                1 - pow(1 - self.gamma, self.number_updates)
            )
        cost_bound = alpha1 * (minimum_cost + alpha0)
        regret_bound = cost_bound - minimum_cost

        if DEBUG:
            """print(
                "  bound for xmin = {:13.6f} xmax = {:13.6f} x0 = {:13.6f}".format(
                    self.inverse_temperature * self.min_cost,
                    self.inverse_temperature * self.max_cost,
                    self.inverse_temperature * J0,
                )
            )"""
            print(
                "  gamma  = {:10.6f}  cost in  [{:10.3f},{:10.3f}]".format(
                    self.gamma, self.min_cost, self.max_cost
                )
            )
            print(
                "  J0     = {:10.6f}  delta   = {:10.6f}  delta0 = {:10.6f}".format(
                    J0, delta, delta0
                )
            )
            print(
                "  alpha1 = {:10.6f}  alpha00 = {:10.6f}  alpha0 = {:10.6f}  alpha1*alpha0 = {:10.6f}".format(
                    alpha1, alpha00, alpha0, alpha0 * alpha1
                )
            )

        if display:
            print(
                "  number_updates = {:13d}  alpha1       = {:13.6f}  alpha0       = {:13.6f}".format(
                    self.number_updates, alpha1, alpha0
                )
            )
            print(
                "  minimum_cost   = {:13.6f}  average_cost = {:13.6f}  cost_bound   = {:13.6f}".format(
                    minimum_cost, average_cost, cost_bound
                )
            )
            print(
                "                                  regret       = {:13.6f}  regret_bound = {:13.6f}".format(
                    regret, regret_bound
                )
            )

        return (
            average_cost,
            minimum_cost,
            cost_bound,
            regret,
            regret_bound,
            self.number_updates,
            alpha1,
            alpha1 * alpha0,
        )
