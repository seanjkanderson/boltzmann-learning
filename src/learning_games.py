#!../venv/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 17:57 2024

@author: Joao Hespanha
"""
from typing import NewType, Any
from collections import OrderedDict
import warnings

import numpy as np
import scipy.stats as sci_stats

from decision_maker import DecisionMaker


Action = NewType("Action", Any)
Measurement = NewType("Measurement", Any)

DEBUG = False


class LearningGame(DecisionMaker):
    """Class to learn no-regret policy"""

    def __init__(
        self,
        action_set: list[Action],
        measurement_set: list[Measurement] = None,
        finite_measurements: bool = True,
        decay_rate: float = 0.0,
        inverse_temperature: float = 0.01,
        seed: int = None,
        time_bound: float = 1.0,
        compute_entropy: bool = True
    ):
        """Constructor for LearningGame class

        Args:
            action_set (list[Action]): list of all possible actions
            measurement_set (list[Measurement], optional): list of all possible measurements.
                Defaults to None.
            finite_measurements (bool, optional): indicates whether the set of measurements is discrete or continuous.
                In the discrete case (True), the measurements will be elements of measurement_set.
                In the continuous case (False), the measurements will be probabilities associated with the classes in
                measurement_set.
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
            time_bound (float, optional): the max time between samples, only used to compute regret, though there are
                other (better) ways to do so in general without this (i.e. compute the regret after one time step).
                Defaults to 1.0.
            compute_entropy (bool, optional): True computes the entropy, False sets it to nan. Entropy for infinite
                measurement case is particularly slow to compute
        """

        # parameters
        self._action_set = action_set
        self.m_actions = len(self._action_set)
        """action_set (set[Action]): set of all possible actions"""

        if measurement_set is None:
            measurement_set = {None}

        self._measurement_set = measurement_set
        """measurement_set (set[Measurement]): set of all possible measurements"""
        # TODO: add check to make sure each list above has no duplicates

        self.finite_measurements = finite_measurements
        """finite_measurements (bool): indicates whether the measurements are finite/discrete or continuous"""

        self.decay_rate = decay_rate
        """decay_rate (float): exponential decay rate for information in units of 1/time.
                Specifically, information decays as
                        exp( -decay_rate * time)
                (0.0 means no decay).
        """

        self.inverse_temperature = inverse_temperature
        """inverse_temperature (float): inverse of thermodynamic temperature for the Boltzmann distribution."""

        # initialize random number generator
        self.rng = np.random.default_rng(seed)
        """rng (np.random._generator.Generator): random number generator to select actions"""

        self.time_bound = time_bound
        """time_bound (float): referred to as \mu_0 in the paper,
                this is the short term bound on the time gap between samples"""

        self.compute_entropy = compute_entropy
        """compute_entropy (bool, optional): True computes the entropy, False sets it to nan. Entropy for infinite
                measurement case is particularly slow to compute"""

        self.reset()

    def reset(self):
        """Reset all parameters: total cost, min/max costs, energies"""

        # initialize energies
        self.energy = OrderedDict(
            [(y, OrderedDict([(a, 0.0) for a in self._action_set])) for y in self._measurement_set]
        )
        """energy (dict[Measurement:dict[Action:float]] 
            self.energy[y][a] (float): energy associated with action a and measurements y,
                Each energy is of the form
                         E[y][a,time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l)) cost[y,a,time_l]
            where t_k is the time at which we got the last update of the energies.
        """

        # initialize time of last update for the energies
        self.time_update = 0
        """time_update (float): time at which the energies were updates last. Only needed when the
        information exponential decay rate `decay_rate` is not zero."""

        # initialize variables needed to compute regret
        self.total_cost = 0.0
        """total_cost (float): total cost incurred so far; used to compute regret"""

        self.normalization_sum = 0
        """normalization_sum (float): sum used to normalize "average cost". Given by
                     W[time_k]=\sum_{l=1}^k  exp(-lambda(time_k-time_l))
        where t_k is the time at which we got the last update of the energies.
        For lambda=0.0 this is just the total number of updates."""

        # initialize bounds
        self.min_cost = np.inf
        """min_cost (float): minimum value of the cost"""
        self.max_cost = -np.inf
        """max_cost (float): maximum value of the cost"""

    def get_Boltzmann_distribution(
        self, measurement: Measurement, time: float = 0.0
    ) -> tuple[np.array, float]:
        """Returns a Boltzmann distribution

        Args:
            measurement (Measurement): measurement, which must be an element of self._measurement_set
            time (float): time for the desired distribution. Defaults to 0.0.

        Returns:
            tuple[np.array,float]:
                probabilities (np.array): probability distribution
                entropy (float): distribution's entropy
        """
        # compute Boltzmann distribution
        # from the equations it may not be immediately clear why the decay is included here. It is because at the time
        # of selecting actions, we need to discount the energy due to time passed since we last updated our energy.
        decay = np.exp(-self.decay_rate * (time - self.time_update))
        if self.finite_measurements:
            energies_array = np.array(
                [decay*v for (k, v) in self.energy[measurement].items()]
            )

        else:
            measurement_values = np.array(list(measurement.values()))
            if np.abs(measurement_values.sum() - 1.) > 1e-6:
                print(measurement.values())
                print(np.sum(list(measurement.values())))
                raise ValueError('Measurement should sum to 1. with precision of 1e-6.')

            # energies_array1 = np.zeros((len(self._action_set), len(self._measurement_set)))
            # for action_idx, action in enumerate(self._action_set):
            #     for measurement_idx, measurement_m in enumerate(self._measurement_set):
            #         energies_array1[action_idx, measurement_idx] = decay*self.energy[measurement_m][action]

            energies_array = np.array(
                [list(self.energy[m].values()) for m in self._measurement_set]
            )
            energies_array = (decay * energies_array).T

        min_energy = np.min(energies_array, axis=0)
        exponent = -self.inverse_temperature * (energies_array - min_energy)
        probabilities = np.exp(exponent)

        if not self.finite_measurements:
            # P(a_k = a, c_k = c | F_k) =
            # exp(-beta * energies_array[a,c]) / \sum_{\bar{a},\bar{c}} exp(-beta * energies_array[\bar{a},\bar{c}])
            pbar_a_c = probabilities / probabilities.sum()
            # P(a_k = a | F_k) = (\sum_c e_c.T@y_k P(a_k = a, c_k = c | F_k)) /
            #                    (\sum_c e_c.T@y_k sum_{\bar{a}} P(a_k = \bar{a}, c_k = c | F_k))
            # probabilities = np.array([pbar_a_c[:, meas_idx]*measurement[meas]
            #                       for meas_idx, meas in enumerate(self._measurement_set)]).sum(axis=0)

            probabilities = (pbar_a_c * measurement_values).sum(axis=1)

        total = np.sum(probabilities)
        if self.finite_measurements:
            # compute entropy, in a way that is safe even if some probabilities become zero
            if self.compute_entropy:
                entropy = -np.dot(probabilities, exponent) / total + np.log(total)
            else:
                entropy = np.nan
            # normalize probability
            probabilities = probabilities / total
        else:
            if total == 0:
                warnings.warn('Probabities sum to 0. Assigning uniform distribution.')
                probabilities += 1.
                total = np.sum(probabilities)
            probabilities = probabilities / total
            if self.compute_entropy:
                entropy = sci_stats.entropy(probabilities)
            else:
                entropy = np.nan
        return probabilities, entropy

    def get_action(
        self, measurement: Measurement, time: float = 0.0, **kwargs
    ) -> tuple[Action, np.array, float]:
        """Gets (optimal) action for a given measurement

        Args:
            measurement (Measurement): measurement, which must be an element of self._measurement_set
            time (float): time for the desired action. Defaults to 0.0.

        Returns:
            tuple[Action,np.array,float]:
                action (Action): (optimal) action
                probabilities (np.array): probability distribution used to select action
                entropy (float): distribution's entropy
        """
        (probabilities, entropy) = self.get_Boltzmann_distribution(measurement, time)
        # get integer
        action_index = self.rng.choice(self.m_actions, p=probabilities)
        # convert to action
        action = list(self._action_set)[action_index]
        return (action, probabilities, entropy)

    def update_energies(
        self, measurement: Measurement, costs: dict[Action, float], time: float = 0.0, **kwargs
    ):
        """Updates energies based on after-the-fact costs

        Args:
            measurement (Measurement): measurement, which must be an element of self._measurement_set
            costs (dict[Action,float]): costs for each action
            time (float): the time at which the measurement is made
        """

        # update bounds
        for k, a in enumerate(self._action_set):
            if costs[a] < self.min_cost:
                self.min_cost = costs[a]
            if costs[a] > self.max_cost:
                self.max_cost = costs[a]

        decay = np.exp(-self.decay_rate * (time - self.time_update))

        # update regrets
        average_cost = 0
        (probabilities, entropy) = self.get_Boltzmann_distribution(measurement, time)
        for k, a in enumerate(self._action_set):
            average_cost += probabilities[k] * costs[a]
        self.total_cost = decay * self.total_cost + average_cost

        for m in self._measurement_set:

            if self.finite_measurements:

                # Update the energies (both continuous and discrete measurements)
                # E[a, time_{k+1}] := \sum_{l=1}^k exp(-lambda(time_k-time_l)) cost[a,time_l]. Let m = k + 1; k = m - 1
                # E[a, time_{m}] = \sum_{l=1}^{m-1} exp(-lambda(time_m-time_l)) cost[a,time_l], m >= 2
                #                   = cost[a,time_{m-1}]exp(-lambda(time_m-time_{m-1}) +
                #                           \sum_{l=1}^{m-2}  exp(-lambda(time_m-time_l)) cost[a,time_l]
                #                   = cost[a,time_{m-1}]exp(-lambda(time_m-time_{m-1}) +
                #                           exp(-lambda(time_{m}-time_{m-1})
                #                           \sum_{l=1}^{m-2}  exp(-lambda(time_{m-1}-time_l)) cost[a,time_l]
                #                   = cost[a,time_{m-1}]exp(-lambda(time_{m-1}-time_{m-2}) +
                #                           exp(-lambda(time_m-time_{m-1}) E[a, time_{m-1}]
                #                   = exp(-lambda(time_m-time_{m-1}) (cost[a, time_{m-1}] + E[a, time_{m-1}])
                if m == measurement:
                    weight = 1.
                else:
                    weight = 0.
            else:
                weight = measurement[m]

            for a in self._action_set:
                self.energy[m][a] = decay * (self.energy[m][a] + costs[a] * weight)

                # Update normalization_sum, which is given by
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

        # decay (depends on t_{k+1} but can be bounded by time_bound)
        inverse_decay = np.exp(self.decay_rate * self.time_bound)

        # compute average cost
        if self.normalization_sum > 0:
            # multiply by inverse_decay in order to get back to paper (13)
            average_cost = inverse_decay * self.total_cost / self.normalization_sum
        else:
            average_cost = 0.0

        # compute average minimum cost
        minimum_cost = 0
        for m in self._measurement_set:
            mn = np.inf
            for a in self._action_set:
                if self.energy[m][a] < mn:
                    # need to multiply by inverse_decay to get back to (13)
                    mn = inverse_decay * self.energy[m][a]
            minimum_cost += mn

        if self.normalization_sum > 0:
            # also need to multiply by inverse_decay to get back to (13)
            minimum_cost *= (inverse_decay / self.normalization_sum)

        regret = average_cost - minimum_cost
        # compute bounds
        J0 = self.min_cost  # TODO: there may be a better choice
        delta = (
            np.exp(self.inverse_temperature * (J0 - self.min_cost))
            - np.exp(self.inverse_temperature * (J0 - self.max_cost))
        ) / (self.inverse_temperature * (self.max_cost - self.min_cost))
        delta0 = (
            (np.exp(self.inverse_temperature * (J0 - self.min_cost)) - 1)
            / self.inverse_temperature
            - J0
            + delta * self.min_cost
        )
        alpha1 = 1 / delta

        if self.finite_measurements:
            cardinality_term = len(self._measurement_set) * np.log(len(self._action_set))
        else:
            cardinality_term = len(self._measurement_set) * np.log(len(self._measurement_set) * len(self._action_set))
        alpha0 = (
            cardinality_term * inverse_decay
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
            alpha1 * alpha0
        )

    def get_energy(self):
        return self.energy


def get_random_integer(rng: np.random, p: np.array) -> int:
    """generate random integer for a given probability distribution

    Args:
        rng (np.random): random number generator
        p (np.array): vector with probabilities, which must add up to 1.0

    Returns:
        int: random integer from 0 to len(p)-1
    """
    cum = np.cumsum(p)
    uniform_rand = rng.random()
    k = np.sum(uniform_rand > cum)
    return k
