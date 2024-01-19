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
        decay_rate: float = 0.0,
        inverse_temperature: float = 0.01,
        seed: int = None,
    ):
        """Constructor for LearningGame class

        Args:
            action_set (set[Action]): set of all possible actions
            measurement_set (set[Measurement], optional): set of all possible measurements.
                Defaults to {None}.
            decay_rate (float, optional): exponential decay rate for information in units of 1/time.
                Specifically, information decays as
                        exp( -decay_rate * time)
                (0.0 means no decay).
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

        self.decay_rate = decay_rate
        """decay_rate (float): exponential decay rate for information in units of 1/time.
                Specifically, information decays as
                        exp( -decay_rate * time)
                (0.0 means no decay).
        """

        self.inverse_temperature = inverse_temperature
        """inverse_temperature (float): inverse of thermodynamic temperature for the Boltzmann distribution."""

        ## initialize random number generator
        self.rng = numpy.random.default_rng(seed)
        """rng (numpy.random._generator.Generator): random number generator to select actions"""

        self.reset()

    def reset(self):
        """Reset all parameters: total cost, min/max costs, energies"""

        ## initialize energies
        self.energy = {
            y: {a: 0.0 for a in self.action_set} for y in self.measurement_set
        }
        """energy (dict[Measurement:dict[Action:float]] 
            self.energy[y][a] (float): energy associated with action a and measurements y,
                Each energy is of the form
                         E[y][a,time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l)) cost[y,a,time_l]
            where t_k is the time at which we got the last update of the energies.
        """

        ## initialize time of last update for the energies
        self.time_update = 0
        """time_update (float): time at which the energies were updates last. Only needed when the
        information exponential decay rate `decay_rate` is not zero."""

        ## initialize variables needed to compute regret
        self.total_cost = 0.0
        """total_cost (float): total cost incurred so far; used to compute regret"""

        self.normalization_sum = 0
        """normalization_sum (float): sum used to normalize "average cost". Given by
                     W[time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l))
        where t_k is the time at which we got the last update of the energies.
        For lambda=0.0 this is just the total number of updates."""

        ## initialize bounds
        self.min_cost = numpy.inf
        """min_cost (float): minimum value of the cost"""
        self.max_cost = -numpy.inf
        """max_cost (float): maximum value of the cost"""

    def get_Boltzmann_distribution(
        self, measurement: Measurement, time: float = 0.0
    ) -> tuple[numpy.array, float]:
        """Returns a Boltzmann distribution

        Args:
            measurement (Measurement): measurement, which must be an element of self.measurement_set
            time (float): time for the desired distribution. Defaults to 0.0.

        Returns:
            tuple[numpy.array,float]:
                probabilities (numpy.array): probability distribution
                entropy (float): distribution's entropy
        """
        # TODO: It seems a little dangerous to rely on set order to construct the probability array
        #        An alternative would be to keep probabilities as a dict, but this would be less efficient

        # compute Boltzmann distribution
        decay = numpy.exp(-self.decay_rate * (time - self.time_update))
        energies_array = numpy.array(
            [decay * v for (k, v) in self.energy[measurement].items()]
        )
        min_energy = numpy.min(energies_array)
        exponent = -self.inverse_temperature * (energies_array - min_energy)
        probabilities = numpy.exp(exponent)
        total = numpy.sum(probabilities)
        # compute entropy, in a way that is safe even if some probabilities become zero
        entropy = -numpy.dot(probabilities, exponent) / total + numpy.log(total)
        # normalize probability
        probabilities = probabilities / total
        return (probabilities, entropy)

    def get_action(
        self, measurement: Measurement, time: float = 0.0
    ) -> tuple[Action, numpy.array, float]:
        """Gets (optimal) action for a given measurement

        Args:
            measurement (Measurement): measurement, which must be an element of self.measurement_set
            time (float): time for the desired action. Defaults to 0.0.

        Returns:
            tuple[Action,numpy.array,float]:
                action (Action): (optimal) action
                probabilities (numpy.array): probability distribution used to select action
                entropy (float): distribution's entropy
        """
        (probabilities, entropy) = self.get_Boltzmann_distribution(measurement, time)
        # get integer
        action_index = get_random_integer(self.rng, probabilities)
        # convert to action
        action = list(self.action_set)[action_index]
        return (action, probabilities, entropy)

    def update_energies(
        self, measurement: Measurement, costs: dict[Action, float], time: float = 0.0
    ):
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

        decay = numpy.exp(-self.decay_rate * (time - self.time_update))

        ## update regrets
        average_cost = 0
        (probabilities, entropy) = self.get_Boltzmann_distribution(measurement, time)
        for k, a in enumerate(self.action_set):
            average_cost += probabilities[k] * costs[a]
        self.total_cost = decay * self.total_cost + average_cost

        ## Update all energies
        #  Each energy is of the form
        #     E[a,time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l)) cost[a,time_l]
        #                = cost[a,time_k] + \sum_{l=1}^{k-1}  exp(-lambda(time_k-time_l)) cost[a,time_l]
        #                = cost[a,time_k] + exp(-lambda(time_k-time_{k-1}))
        #                                       \sum_{l=1}^{k-1}  exp(-lambda(time_{k-1}-time_l)) cost[a,time_l]
        #                = cost[a,time_k] + exp(-lambda(time_k-time_{k-1})) E[a,time_{k-1}]
        for m in self.measurement_set:
            if m == measurement:
                for a in self.action_set:
                    self.energy[m][a] = decay * self.energy[m][a] + costs[a]
            else:
                for a in self.action_set:
                    self.energy[m][a] = decay * self.energy[m][a]

        ## Update normalization_sum, which is given by
        #     W[time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l))
        #              = 1 + \sum_{l=1}^{k-1}  exp(-lambda(time_k-time_l))
        #              = 1 + exp(-lambda(time_k-time_{k-1}))\sum_{l=1}^{k-1}  exp(-lambda(time_{k-1}-time_l))
        #              = 1 + exp(-lambda(time_k-time_{k-1})) W[time_{k-1}]
        self.normalization_sum = decay * self.normalization_sum + 1

        # Update time
        self.time_update = time

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
                normalization_sum (float): sum used to normalize "average cost".
                alpha1 (float): multiplicative factor for theoretical bound
                alpha0 (float): additive factor for theoretical bound

        """

        ## compute average cost
        if self.normalization_sum > 0:
            average_cost = self.total_cost / self.normalization_sum
        else:
            average_cost = 0.0

        ## compute average minimum cost
        minimum_cost = 0
        for m in self.measurement_set:
            mn = numpy.inf
            for a in self.action_set:
                if self.energy[m][a] < mn:
                    mn = self.energy[m][a]
            minimum_cost += mn

        if self.normalization_sum > 0:
            minimum_cost /= self.normalization_sum

        regret = average_cost - minimum_cost

        ## compute bounds
        J0 = self.min_cost  # TODO: there may be a better choice
        J0 = (self.min_cost + self.max_cost) / 2  # TODO: there may be a better choice
        delta = (
            numpy.exp(self.inverse_temperature * (J0 - self.min_cost))
            - numpy.exp(self.inverse_temperature * (J0 - self.max_cost))
        ) / (self.inverse_temperature * (self.max_cost - self.min_cost))
        delta0 = (
            (numpy.exp(self.inverse_temperature * (J0 - self.min_cost)) - 1)
            / self.inverse_temperature
            - J0
            + delta * self.min_cost
        )
        alpha1 = 1 / delta

        alpha0 = (
            len(self.measurement_set)
            * numpy.log(len(self.action_set))
            / self.inverse_temperature
            / self.normalization_sum
        ) + delta0

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
                "  alpha1 = {:10.6f}  alpha0 = {:10.6f}  alpha1*alpha0 = {:10.6f}".format(
                    alpha1, alpha0, alpha0 * alpha1
                )
            )

        if display:
            print(
                "  normalization_sum = {:13.6f}  alpha1       = {:13.6f}  alpha0       = {:13.6f}".format(
                    self.normalization_sum, alpha1, alpha0
                )
            )
            print(
                "  minimum_cost      = {:13.6f}  average_cost = {:13.6f}  cost_bound   = {:13.6f}".format(
                    minimum_cost, average_cost, cost_bound
                )
            )
            print(
                "                                     regret       = {:13.6f}  regret_bound = {:13.6f}".format(
                    regret, regret_bound
                )
            )

        return (
            average_cost,
            minimum_cost,
            cost_bound,
            regret,
            regret_bound,
            self.normalization_sum,
            alpha1,
            alpha1 * alpha0,
        )
