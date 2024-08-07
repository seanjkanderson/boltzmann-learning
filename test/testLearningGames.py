import unittest
import numpy

import sys

sys.path.append("src")
import LearningGames


class Test010RandomIntegers(unittest.TestCase):
    """TestRandomIntegers Class to test random number generator"""

    def test010_get_random_integer1(self):
        print("\ntest_get_random_integer1:")
        rng = numpy.random.default_rng(5)
        p = [3]

        M = 10000

        ks = [LearningGames.get_random_integer(rng, p) for i in range(M)]
        histogram = numpy.zeros(len(p))
        for k in ks:
            histogram[k] += 1
        histogram /= M
        err = histogram - numpy.array(p) / sum(p)
        print("  actual distribution:    ", p)
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)
        print("  error:                  ", err)
        print("  rms error:              ", numpy.linalg.norm(err) / M)

        self.assertLess(numpy.linalg.norm(err) / M, 1e-5)

    def test011_get_random_integer4(self):
        print("\ntest_get_random_integer4:")
        rng = numpy.random.default_rng(5)
        p = [0.2, 0.1, 0.5, 0.2]

        M = 10000

        ks = [LearningGames.get_random_integer(rng, p) for i in range(M)]
        histogram = numpy.zeros(len(p))
        for k in ks:
            histogram[k] += 1
        histogram /= M
        err = histogram - numpy.array(p) / sum(p)
        print("  actual distribution:    ", p)
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)
        print("  error:                  ", err)
        print("  rms error:              ", numpy.linalg.norm(err) / M)

        self.assertLess(numpy.linalg.norm(err) / M, 1e-5)

    def test012_get_random_integer5(self):
        print("\ntest_get_random_integer5:")
        rng = numpy.random.default_rng(5)
        p = [0.2, 0.1, 0.5, 0.2, 0.0]

        M = 10000

        ks = [LearningGames.get_random_integer(rng, p) for i in range(M)]
        histogram = numpy.zeros(len(p))
        for k in ks:
            histogram[k] += 1
        histogram /= M
        err = histogram - numpy.array(p) / sum(p)
        print("  actual distribution:    ", p)
        print("  number of samples:      ", M)
        print("  empirical distribution: ", histogram)
        print("  error:                  ", err)
        print("  rms error:              ", numpy.linalg.norm(err) / M)

        self.assertLess(numpy.linalg.norm(err) / M, 1e-5)


class Test020LearningGames(unittest.TestCase):
    """TestLearningGames Class to test LearningGames constructor"""

    def test020_LearningGames(self):
        print("\ntest_LearningGames:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        lg = LearningGames.LearningGame(action_set, measurement_set, seed=0)
        print(vars(lg))

        # Check object variables
        self.assertEqual(lg.action_set, action_set)
        self.assertEqual(lg.measurement_set, measurement_set)
        self.assertEqual(lg.min_cost, +numpy.inf)
        self.assertEqual(lg.max_cost, -numpy.inf)
        self.assertEqual(lg.inverse_temperature, 0.01)
        self.assertEqual(len(lg.energy), len(measurement_set))
        for m in measurement_set:
            self.assertEqual(len(lg.energy[m]), len(action_set))

    def test021_get_action(self):
        print("\ntest_get_action:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        lg = LearningGames.LearningGame(action_set, measurement_set, seed=0)

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
        print("  rms error:              ", numpy.linalg.norm(err) / M)

        self.assertEqual(len(err), len(action_set))
        self.assertLess(numpy.linalg.norm(err) / M, 1e-5)

    def test022_update_energies(self):
        print("\ntest_update_energies:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast
        lg = LearningGames.LearningGame(
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
        lg = LearningGames.LearningGame(
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
        print("\ntest023_regret_gamma:")
        action_set = {"a1", "a2", "a3"}
        measurement_set = {"y1", "y2"}
        # use low temperature to converge to deterministic very fast, but bound is bad
        lg = LearningGames.LearningGame(
            action_set, measurement_set, inverse_temperature=0.01, seed=0
        )

        M = 20000
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
