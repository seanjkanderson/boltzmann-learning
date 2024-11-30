import itertools
from collections import OrderedDict

import unittest
import numpy as np
import numpy.testing as nptest

from learning_games import LearningGame


class Test020LearningGames(unittest.TestCase):
    """TestLearningGames Class to test LearningGames constructor"""

    def test020_LearningGames(self):
        print("\ntest_LearningGames:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        lg = LearningGame(action_set, measurement_set, seed=0)
        print(vars(lg))

        # Check object variables
        self.assertEqual(lg._action_set, action_set)
        self.assertEqual(lg._measurement_set, measurement_set)
        self.assertEqual(lg.min_cost, +np.inf)
        self.assertEqual(lg.max_cost, -np.inf)
        self.assertEqual(lg.inverse_temperature, 0.01)
        self.assertEqual(len(lg.energy), len(measurement_set))
        for m in measurement_set:
            self.assertEqual(len(lg.energy[m]), len(action_set))

    def test021_get_action(self):
        print("\ntest_get_action:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        lg = LearningGame(action_set, measurement_set, seed=0)

        # Check distribution from get_action (starts as uniform)
        M = 10000
        As = [lg.get_action("y1")[0] for i in range(M)]
        histogram = {k: 0 for k in action_set}
        for a in As:
            histogram[a] += 1
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)
        err = [v / M - 1 / len(action_set) for (k, v) in histogram.items()]
        print("  error:                  ", err)
        print("  rms error:              ", np.linalg.norm(err) / M)

        self.assertEqual(len(err), len(action_set))
        self.assertLess(np.linalg.norm(err) / M, 1e-5)

    def test022_update_energies(self):
        print("\ntest_update_energies:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast
        lg = LearningGame(
            action_set, measurement_set, inverse_temperature=10.0, seed=0
        )

        # with low temperature, all probability goes to lowest energy
        reward = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        lg.update_energies("y1", reward)
        print(vars(lg))

        # regret: before update all actions have equal probability
        self.assertAlmostEqual(lg.total_cost, 2.0)

        # new distribution: check distribution from get_action (starts as uniform)
        M = 10000
        As = [lg.get_action("y1")[0] for i in range(M)]
        histogram = {k: 0 for k in action_set}
        for a in As:
            histogram[a] += 1
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)

        self.assertGreater(histogram["a1"], M - 2)
        self.assertLess(histogram["a2"], 2)
        self.assertLess(histogram["a3"], 2)

    def test023_regret_T(self):
        print("\ntest023_regret_T:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast, but bound is bad
        lg = LearningGame(
            action_set, measurement_set, inverse_temperature=0.01, seed=0
        )

        M = 10000
        reward1 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        reward2 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        for i in range(M):
            lg.update_energies("y1", reward1)
            lg.update_energies("y2", reward2)
        # print(vars(lg))

        (
            average_cost,
            minimum_cost,
            cost_bound,
            regret,
            regret_bound,
            steps,
            alpha1,
            alpha0,
        ) = lg.get_regret(display=True)

        self.assertEqual(lg.min_cost, 1.0)
        self.assertEqual(lg.max_cost, 3.0)
        self.assertLess(average_cost, cost_bound)
        self.assertLess(regret, regret_bound)

    def test024_regret_gamma(self):
        # TODO: rename this test as gamma is no longer a parameter?
        print("\ntest023_regret_gamma:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast, but bound is bad
        lg = LearningGame(
            action_set, measurement_set, inverse_temperature=0.01, seed=0
        )

        M = 20000
        reward1 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        reward2 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        for i in range(M):
            lg.update_energies("y1", reward1)
            lg.update_energies("y2", reward2)
        (
            average_cost,
            minimum_cost,
            cost_bound,
            regret,
            regret_bound,
            steps,
            alpha1,
            alpha0,
        ) = lg.get_regret(display=True)

        self.assertEqual(lg.min_cost, 1.0)
        self.assertEqual(lg.max_cost, 3.0)
        self.assertLess(average_cost, cost_bound)
        self.assertLess(regret, regret_bound)


class TestContinuousMeasurementLearningGames(unittest.TestCase):
    """Test LearningGames in the continuous measurement setting"""
    # TODO: likely can combine this with test for finite measurement class with proper use of setUp/tearDown
    def test_get_action(self):
        """Check distribution from get_action (starts as uniform)"""
        print("\ntest_get_action:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        lg = LearningGame(action_set, measurement_set, seed=0,
                                        finite_measurements=False)

        M = 10000
        measurement = {"y1": 0.5, "y2": 0.5}
        As = [lg.get_action(measurement)[0] for _ in range(M)]
        histogram = {k: 0 for k in action_set}
        for a in As:
            histogram[a] += 1
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)
        err = [v / M - 1 / len(action_set) for (k, v) in histogram.items()]
        print("  error:                  ", err)
        print("  rms error:              ", np.linalg.norm(err) / M)

        self.assertEqual(len(err), len(action_set))
        self.assertLess(np.linalg.norm(err) / M, 1e-5)

    def test_update_energies(self):
        print("\ntest_update_energies:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast
        lg = LearningGame(
            action_set, measurement_set, inverse_temperature=10.0, seed=0,
            finite_measurements=False
        )

        # with low temperature, all probability goes to lowest energy
        reward = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        measurement = {"y1": 1., "y2": 0.}
        lg.update_energies(measurement, reward)
        print(vars(lg))

        # regret: before update all actions have equal probability
        self.assertAlmostEqual(lg.total_cost, 2.0)

        # new distribution: check distribution from get_action (starts as uniform)
        M = 10000
        As = [lg.get_action(measurement)[0] for _ in range(M)]
        histogram = {k: 0 for k in action_set}
        for a in As:
            histogram[a] += 1
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)

        self.assertGreater(histogram["a1"], M - 2)
        self.assertLess(histogram["a2"], 2)
        self.assertLess(histogram["a3"], 2)

    def test_regret_T(self):
        print("\ntest023_regret_T:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast, but bound is bad
        lg = LearningGame(
            action_set, measurement_set, inverse_temperature=0.01, seed=0,
            finite_measurements=False
        )

        M = 10000
        reward1 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        reward2 = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        for _ in range(M):
            measurement_y1 = {"y1": 1., "y2": 0.}
            lg.update_energies(measurement_y1, reward1)
            measurement_y2 = {"y1": 0., "y2": 1.}
            lg.update_energies(measurement_y2, reward2)
        (
            average_cost,
            minimum_cost,
            cost_bound,
            regret,
            regret_bound,
            steps,
            alpha1,
            alpha0,
        ) = lg.get_regret(display=True)

        self.assertEqual(lg.min_cost, 1.0)
        self.assertEqual(lg.max_cost, 3.0)
        self.assertLess(average_cost, cost_bound)
        self.assertLess(regret, regret_bound)

    def test_extrema_bound_equivalence(self):
        action_set = ["a1", "a2", "a3"]
        measurement_set = ["y1", "y2"]
        # use low temperature to converge to deterministic very fast, but bound is bad
        beta = 1.
        decay_rate = 0.

        M = 20000
        all_costs = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        raw_measurement = None
        expected_costs = np.zeros((M,))
        v1 = np.zeros((len(measurement_set),))
        v2 = np.zeros((len(measurement_set),))
        v1[0] = 1.
        v2[1] = 1.
        vmax = 1.
        dt = 1.
        decay = np.exp(-decay_rate*dt)
        measurement_extrema = [OrderedDict(y1=1., y2=0.), OrderedDict(y1=0., y2=1.)]
        v_options = [v2, v1]
        for stat_a, (measurement, measurement_fin), v in itertools.product(action_set, zip(measurement_extrema, measurement_set), v_options):
            print(stat_a, measurement, measurement_fin, v)
            lg_fin = LearningGame(
                action_set, measurement_set, inverse_temperature=beta, seed=0, decay_rate=decay_rate,
                finite_measurements=True
            )

            lg_inf = LearningGame(
                action_set, measurement_set, inverse_temperature=beta, seed=0, decay_rate=decay_rate,
                finite_measurements=False
            )
            lhs_running_cost = 0.
            rhs_running_cost = 0.
            rhs_running_cost_fin = 0.
            rhs_running_inner_product = 0.
            rhs_running_indicator = 0.

            for idx in range(M):
                action, prob, _ = lg_inf.get_action(measurement=measurement, time=idx,
                                                                       raw_measurement=raw_measurement)

                expected_costs[idx] = (np.array([val for _, val in all_costs.items()]) * prob).sum()
                # Learn
                lg_inf.update_energies(measurement=measurement, costs=all_costs, action=action,
                                               raw_measurement=raw_measurement,
                                               time=idx)
                (_, _, _, _, _, _, alpha1, _) = lg_inf.get_regret()
                delta = 1/alpha1

                inner_product_v_y = (np.array([val for _, val in measurement.items()]) * v).sum()
                #update the LHS
                lhs_running_cost = decay * (inner_product_v_y * expected_costs[idx] + lhs_running_cost)

                lhs = beta*delta*lhs_running_cost
                # update the RHS
                rhs_running_inner_product = decay * (inner_product_v_y + rhs_running_inner_product)
                rhs_running_cost = decay * (inner_product_v_y * all_costs[stat_a] + rhs_running_cost)
                additive_term = vmax*np.log(len(action_set)*len(measurement_set)) - beta*(1-delta)*lg_inf.min_cost*rhs_running_inner_product
                rhs = beta * rhs_running_cost + additive_term

                self.assertLessEqual(lhs, rhs)

                # update the finite measurement case TODO: should be terser way to construct this comparison
                action_fin, prob_fin, _ = lg_fin.get_action(measurement=measurement_fin, time=idx,
                                                raw_measurement=raw_measurement)

                expected_costs[idx] = (np.array([v for k, v in all_costs.items()]) * prob_fin).sum()
                # Learn
                lg_fin.update_energies(measurement=measurement_fin, costs=all_costs, action=action_fin,
                                   raw_measurement=raw_measurement,
                                   time=idx)
                (_, _, _, _, _, _, alpha1, _) = lg_fin.get_regret()
                delta_fin = 1 / alpha1

                # update the LHS
                lhs_fin = beta * delta_fin * lg_fin.total_cost
                # update the RHS
                # we only measure a particular measurement the whole time
                rhs_running_indicator = decay * (1. + rhs_running_indicator)
                rhs_running_cost_fin = decay * (all_costs[stat_a] + rhs_running_cost_fin)
                additive_term_fin = np.log(len(action_set)) - beta * (
                        1 - delta) * lg_fin.min_cost * rhs_running_indicator
                rhs_fin = beta * rhs_running_cost_fin + additive_term_fin

                self.assertLessEqual(lhs_fin, rhs_fin)
                self.assertIsNone(nptest.assert_almost_equal(prob_fin, prob))  # TODO: cleaner if use nosetests in future

                # compare the energies for the two
                for (key_fin, val_fin), (key, val) in zip(lg_fin.energy.items(), lg_inf.energy.items()):
                    self.assertEqual(key_fin, key)  # ordering should be consistent in the ordered dicts
                    self.assertEqual(val_fin, val)  # the energies should be the same

    def test_form_of_optimal_stationary_stochastic_policy(self):
        pass

    def test_theorem_2(self):
        """checks to make sure that the inequality in Theorem 2 holds."""
        action_set = ["a1", "a2", "a3"]
        measurement_set = ["y1", "y2"]
        # use low temperature to converge to deterministic very fast, but bound is bad
        beta = 1.
        decay_rate = 0.

        M = 20000
        all_costs = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        raw_measurement = None
        expected_costs = np.zeros((M,))
        e_vecs = np.eye(len(measurement_set))
        # because it's an LP, the min/max occur on the boundaries of the feasible region.
        dt = 1.
        decay = np.exp(-decay_rate * dt)
        measurement_tests = [OrderedDict(y1=0.5, y2=0.5), OrderedDict(y1=0.9, y2=0.1),
                               OrderedDict(y1=0.99, y2=0.01), OrderedDict(y1=0.01, y2=0.99)]

        # optimal policy is to always play action 1 given the cost structure. TODO: stochastic example?
        z = dict(a1=np.array([[1., 1.]]), a2=np.array([[0., 0.]]), a3=np.array([[0., 0.]]))

        for stat_a, measurement, e_c in itertools.product(action_set, measurement_tests, e_vecs):
            print(stat_a, measurement, z, e_c)

            lg_inf = LearningGame(
                action_set, measurement_set, inverse_temperature=beta, seed=0, decay_rate=decay_rate,
                finite_measurements=False
            )

            rhs_running_inner_product = 0.
            for idx in range(M):
                action, prob, _ = lg_inf.get_action(measurement=measurement, time=idx,
                                                    raw_measurement=raw_measurement)

                expected_costs[idx] = (np.array([val for _, val in all_costs.items()]) * prob).sum()
                # Learn
                lg_inf.update_energies(measurement=measurement, costs=all_costs, action=action,
                                       raw_measurement=raw_measurement,
                                       time=idx)
                w_k = lg_inf.normalization_sum
                if w_k == 0:
                    w_k = 1.

                (_, _, _, _, _, _, alpha1, _) = lg_inf.get_regret()
                delta = 1 / alpha1

                # update the LHS
                lhs = lg_inf.total_cost / w_k
                # update the RHS
                # first make sure that z_a satisfies constraints
                inner_prods = list((np.array([val for _, val in measurement.items()]) *
                     z[a]).sum()for a in action_set)
                self.assertEqual(np.sum(inner_prods), 1.0)
                [self.assertGreaterEqual(elem, 0.) for elem in inner_prods]
                # update the RHS costs
                minimum_expected_cost = sum((np.array([val for _, val in measurement.items()]) *
                                          z[a]).sum()*all_costs[a] for a in action_set)
                rhs_running_inner_product = decay * (minimum_expected_cost + rhs_running_inner_product)
                rhs_weighted_cost = (1 / w_k) * rhs_running_inner_product
                alpha0 = len(measurement_set)*np.log(len(measurement_set)*len(action_set)) / (beta * w_k) - \
                                (1 - delta) * lg_inf.min_cost
                rhs = alpha1 * (rhs_weighted_cost + alpha0)
                # make sure that the inequality in Theorem 2 holds
                self.assertLessEqual(lhs, rhs)

    def test_nonextrema_bound(self):
        # TODO: remove? seems to be essentially the same as theorem 2 bound (above)
        action_set = ["a1", "a2", "a3"]
        measurement_set = ["y1", "y2"]
        beta = 1.
        decay_rate = 0.

        M = 20000
        all_costs = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        raw_measurement = None
        expected_costs = np.zeros((M,))
        v1 = np.zeros((len(measurement_set),))
        v2 = np.zeros((len(measurement_set),))

        v1[0] = 1.
        v2[1] = 1.
        vmax = 1.
        dt = 1.
        decay = np.exp(-decay_rate * dt)

        measurement_extrema = [OrderedDict(y1=0.5, y2=0.5), OrderedDict(y1=0.6, y2=0.4), OrderedDict(y1=0.8, y2=0.2),
                               OrderedDict(y1=0.9, y2=0.1),
                               OrderedDict(y1=0.99, y2=0.01), OrderedDict(y1=0.01, y2=0.99)]
        v_options = [v2, v1]
        for stat_a, measurement, v in itertools.product(action_set, measurement_extrema, v_options):
            print(stat_a, measurement, v)

            lg_inf = LearningGame(
                action_set, measurement_set, inverse_temperature=beta, seed=0, decay_rate=decay_rate,
                finite_measurements=False
            )
            lhs_running_cost = 0.
            rhs_running_cost = 0.
            rhs_running_inner_product = 0.

            for idx in range(M):
                action, prob, _ = lg_inf.get_action(measurement=measurement, time=idx,
                                                    raw_measurement=raw_measurement)

                expected_costs[idx] = (np.array([val for _, val in all_costs.items()]) * prob).sum()
                # Learn
                lg_inf.update_energies(measurement=measurement, costs=all_costs, action=action,
                                       raw_measurement=raw_measurement,
                                       time=idx)
                (_, _, _, _, _, _, alpha1, _) = lg_inf.get_regret()
                delta = 1 / alpha1

                inner_product_v_y = (np.array([val for _, val in measurement.items()]) * v).sum()
                # update the LHS
                lhs_running_cost = decay * (inner_product_v_y * expected_costs[idx] + lhs_running_cost)

                lhs = beta * delta * lhs_running_cost
                # update the RHS
                rhs_running_inner_product = decay * (inner_product_v_y + rhs_running_inner_product)
                rhs_running_cost = decay * (inner_product_v_y * all_costs[stat_a] + rhs_running_cost)
                additive_term = vmax * np.log(len(action_set)*len(measurement_set)) - beta * (
                            1 - delta) * lg_inf.min_cost * rhs_running_inner_product
                rhs = beta * rhs_running_cost + additive_term

                self.assertLessEqual(lhs, rhs)

    def test_sum_actions_bound(self):
        action_set = ["a1", "a2", "a3"]
        measurement_set = ["y1", "y2"]

        beta = 1.
        decay_rate = 0.

        M = 20000
        all_costs = {"a1": 1.0, "a2": 2.0, "a3": 3.0}
        raw_measurement = None
        expected_costs = np.zeros((M,))
        v1 = np.ones((len(measurement_set),))
        v2 = np.zeros((len(measurement_set),))
        v3 = v2
        z = {action_set[0]: v1, action_set[1]: v2, action_set[2]: v3}
        vmax = [np.max(z[a]) for a in action_set]
        dt = 1.
        decay = np.exp(-decay_rate * dt)
        measurement_testpoints = [OrderedDict(y1=1., y2=0.), OrderedDict(y1=0., y2=1.), OrderedDict(y1=0.5, y2=0.5)]
        v_options = [v1]
        for measurement, v in itertools.product(measurement_testpoints, v_options):
            print(measurement, v)
            yk = np.array([val for _, val in measurement.items()])

            lg_inf = LearningGame(
                action_set, measurement_set, inverse_temperature=beta, seed=0, decay_rate=decay_rate,
                finite_measurements=False
            )
            lhs_running_cost = 0.
            rhs_running_cost = 0.
            rhs_running_inner_product = 0.

            for idx in range(M):
                action, prob, _ = lg_inf.get_action(measurement=measurement, time=idx,
                                                    raw_measurement=raw_measurement)

                expected_costs[idx] = (np.array([val for _, val in all_costs.items()]) * prob).sum()
                # Learn
                lg_inf.update_energies(measurement=measurement, costs=all_costs, action=action,
                                       raw_measurement=raw_measurement,
                                       time=idx)
                (_, _, _, _, _, _, alpha1, _) = lg_inf.get_regret()
                delta = 1 / alpha1

                inner_product_v_y = (np.array([val for _, val in measurement.items()]) * v).sum()
                # update the LHS
                lhs_running_cost = decay * (inner_product_v_y * expected_costs[idx] + lhs_running_cost)

                lhs = beta * delta * lhs_running_cost
                # update the RHS
                weighted_costs = np.sum([z[a].T*yk*all_costs[a] for a in action_set])
                rhs_running_cost = decay * (weighted_costs + rhs_running_cost)
                sum_infty_norm = np.sum(vmax)
                rhs_running_inner_product = decay * (inner_product_v_y + rhs_running_inner_product)
                additive_term = sum_infty_norm * np.log(len(action_set)*len(measurement_set)) - beta * (
                            1 - delta) * lg_inf.min_cost * rhs_running_inner_product
                rhs = beta * rhs_running_cost + additive_term

                self.assertLessEqual(lhs, rhs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
